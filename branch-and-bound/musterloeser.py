"""Implements a simple branch-and-bound algorithm with the help of Pyomo.
By default the solver returns the search tree which can be used to visulize
the algortihm.

Author: Marvin Meck (https://github.com/marvin-meck/)
"""

import logging
import re
import numpy as np
import pyomo.environ as pyo
import igraph as ig

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

logger = logging.getLogger("musterloeser")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = logging.FileHandler("./musterloeser.log")
fh.setLevel(logging.DEBUG)

logger.addHandler(ch)
logger.addHandler(fh)


def max_fractional_rule(root: pyo.ConcreteModel, sub: pyo.ConcreteModel) -> str:
    """
    returns the variable name that is the most fractional
    """

    arrlen = 0
    strlen = 0
    for vname in root.component_map(pyo.Var):
        var = getattr(root, vname)
        for idx in var.keys():
            if var[idx].is_integer or var[idx].is_binary:
                arrlen += 1
            if strlen < len(var[idx].name):
                strlen = len(var[idx].name)

    logger.debug(f"processing fractionals of {arrlen} variables...")

    var_list = np.empty(arrlen, dtype=np.dtype(f"U{strlen}"))
    val_list = np.empty(arrlen)

    i = 0
    for vname in root.component_map(pyo.Var):
        var = getattr(root, vname)
        rlx = getattr(sub, vname)
        for idx in var.keys():
            if not var[idx].is_continuous():
                logger.debug("{} = {}".format(rlx[idx].name, rlx[idx].value))
                var_list[i] = rlx[idx].name
                val_list[i] = abs(0.5 - (rlx[idx].value % 1))
                i += 1

    i = np.argmin(val_list)
    logger.debug("Done! Selected variable {}.".format(var_list[i]))

    m = re.search(r"\A(\w+)\[[\w,]+\]\Z", var_list[i])
    vname = m.group(1)

    prog = re.compile(r"(\w+)[\,|\]]")
    index = [match.group(1) for match in prog.finditer(var_list[i])]

    for num, val in enumerate(index):
        index[num] = int(val) if val.isnumeric() else val

    return vname, tuple(index)


def lifo_branching(
    root, num_nodes, node_list, var_selection_rule, graph: ig.Graph = None
):

    prob, _ = node_list.pop()
    name = prob.getname()
    logger.debug(f"popped {name} from stack")

    vname, index = var_selection_rule(root, prob)

    def branch_floor_rule(model):
        var = getattr(model, vname)
        return var[index] <= np.floor(var[index].value)

    def branch_ceil_rule(model):
        var = getattr(model, vname)
        return var[index] >= np.ceil(var[index].value)

    sub_1, sub_2 = prob.clone(), prob.clone()
    sub_1.name = "P{}".format(num_nodes + 1)
    sub_2.name = "P{}".format(num_nodes + 2)

    sub_1.add_component(
        f"{sub_1.name}_branch_{vname}_{index}_floor",
        pyo.Constraint(rule=branch_floor_rule),
    )
    sub_2.add_component(
        f"{sub_2.name}_branch_{vname}_{index}_ceil",
        pyo.Constraint(rule=branch_ceil_rule),
    )

    graph.add_vertices(2)
    p0, p1, p2 = graph.vs.find(label=prob.name), graph.vs[-2], graph.vs[-1]
    p1["label"] = sub_1.name
    p2["label"] = sub_2.name

    graph.add_edges([(p0, p1), (p0, p2)])
    e1, e2 = graph.es[-2], graph.es[-1]

    e1["xlabel"] = "{}[{}] <= {}".format(
        vname, index, np.floor(getattr(sub_1, vname)[index].value)
    )
    e2["xlabel"] = "{}[{}] >= {}".format(
        vname, index, np.ceil(getattr(sub_2, vname)[index].value)
    )

    return [sub_1, sub_2]


def is_feasible(root, sub, feas_tol=1e-6):
    """
    is sub feasible on root?
    """
    for name in sub.component_map(pyo.Var):
        root_var = getattr(root, name)
        sub_var = getattr(sub, name)

        for idx in sub_var.keys():
            if root_var[idx].is_continuous():
                val = sub_var[idx].value
            else:
                val = np.round(sub_var[idx].value / feas_tol) * feas_tol
                if not np.abs(val % 1) <= feas_tol:
                    return 0

            root_var[idx].set_value(val)

    return 1


rel_gap = lambda incumbent, dual: abs(incumbent - dual) / min(incumbent, dual)


def branch_and_bound(
    root: pyo.ConcreteModel,
    lp_solver: str = "glpk",
    var_selection_rule=None,
    branching_rule=None,
    log_tree=True,
) -> ig.Graph:
    """A branch-and-bound algorithm."""

    print(f"okay let's go!")

    if lp_solver == "gurobi":
        solver = pyo.SolverFactory(lp_solver, solver_io="python")
    else:
        solver = pyo.SolverFactory(lp_solver)

    if var_selection_rule is None:
        var_selection_rule = max_fractional_rule
        logger.info(f"variable selection rule is 'max fractional rule'")

    if branching_rule is None:
        branching_rule = lifo_branching
        logger.info(f"branching rule is 'LIFO'")

    # solve LP-Relaxation
    p0 = root.clone()
    p0.name = "P0"
    pyo.TransformationFactory("core.relax_integer_vars").apply_to(p0)
    results = solver.solve(p0)

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise (
            f"root relaxation cannot be solved! Ended with solver status {results.solver.status}"
        )

    dual_bound = results.problem.upper_bound  # note: must equal lower_bound
    print(f"solved root problem: {dual_bound}\n")

    # create candidate list
    node_list = [(p0, dual_bound)]
    incumbent = np.nan
    node = 0
    if log_tree:
        g = ig.Graph()
        g.add_vertices(1)
        g.vs[-1]["label"] = p0.name
        g.vs[-1]["xlabel"] = "DB = {:.2}".format(dual_bound)
    else:
        g = None

    print("node | incumbent |      dual |  gap")
    line = "{:4d} | {:9.2f} | {:9.2f} | {:4.2f}".format(
        node, incumbent, dual_bound, rel_gap(incumbent, dual_bound)
    )
    print(line)
    while len(node_list):

        # select candidate from list, remove and branch
        sub_list = branching_rule(root, node, node_list, var_selection_rule, g)
        for sub in sub_list:
            node += 1
            results = solver.solve(sub)  # solve relaxation

            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                optimum = results.problem.upper_bound
                logger.debug(
                    "{} is feasible with local optimum: {}, dual: {}".format(
                        sub.name, optimum, dual_bound
                    )
                )

                if is_feasible(root, sub):
                    logger.debug("solution is feasible on original problem...")
                    if np.isnan(incumbent) or (incumbent < optimum):
                        logger.debug("new incumbent")
                        incumbent = optimum
                        for num, (_, db) in enumerate(node_list):
                            if db < incumbent:
                                del node_list[num]
                        if log_tree:
                            g.vs.find(label=sub.name)["xlabel"] = f"incumbent"
                    else:
                        logger.debug("sub-optimal")
                        if log_tree:
                            g.vs.find(label=sub.name)["xlabel"] = f"sub-optimal"

                elif np.isnan(incumbent) or (optimum > incumbent):  # todo: case min
                    logger.debug("adding candidate {}".format(sub.name))
                    node_list.append((sub, optimum))
                    if log_tree:
                        g.vs.find(label=sub.name)["xlabel"] = "DB = {:.2}".format(
                            optimum
                        )
                else:
                    logger.debug(
                        "dropping candidate {} with DB = {:.2} <= {:.2}".format(
                            sub.name, optimum, incumbent
                        )
                    )
                    if log_tree:
                        g.vs.find(label=sub.name)["xlabel"] = (
                            "DB = {:.2} <= {:.2}".format(optimum, incumbent)
                        )

            elif (
                results.solver.termination_condition
                == pyo.TerminationCondition.infeasible
            ):
                logger.debug("{} is infeasible!".format(sub.name))
                if log_tree:
                    g.vs.find(label=sub.name)["xlabel"] = f"infeasible"
            else:
                raise AssertionError("not reachable")

            tmp = [db for _, db in node_list]
            tmp.append(incumbent)
            dual_bound = max(tmp)  # todo: case min
            gap = rel_gap(incumbent, dual_bound)
            line = "{:4d} | {:9.2f} | {:9.2f} | {:4.2f}".format(
                node, incumbent, dual_bound, gap
            )
            print(line)

    print("solved!")
    print(f"primal: {incumbent}\ndual: {dual_bound}")

    return g


if __name__ == "__main__":
    pass
