"""Minimum volume problem from truss topology optimization (Pyomo model)

The minimum volume problem aims to find the optimal layout (topology) and
cross-sectional areas of truss members that minimize the total structural
volume (and hence, typically the material usage and cost).

The linear programming formulation (LP) of the problem is to minimize the sum
of the cross-sectional areas of the truss members within the ground structure
multiplied by their lengths, thereby finding the minimum total volume of the
structure. The model considers static equilibrium between the horizontal and
vertical bar forces and the applied loads. The bar forces are constrained by
stress limits for tension and compression. The topological decisions arise from
the separation of variables into basic and non-basic variables, i.e., bars
associated with non-basic variables (zero-force members) can be eliminated from
the ground structure [1].

The LP is taken from Oberndorfer et al. [1] and implemented using the 
Pyomo modeling language [2].

Author: Marvin Meck (https://github.com/marvin-meck/)

References:
-----------

[1]  J. M. Oberndorfer, W. Achtziger, and Hörnlein, H. R. E. M., “Two
     approaches for truss topology optimization: a comparison for practical
     use,” Structural Optimization, vol. 11, no. 3–4, pp. 137–144, Jan. 1996,
     doi: 10.1007/BF01197027.

[2]  W. E. Hart, J.-P. Watson, C. D. Laird, B. L. Nicholson, and J. D. Siirola,
     Pyomo: Optimization Modeling in Python, Second edition / William E Hart
     [and six others]., vol. 67. in Springer Optimization and Its Applications,
     vol. 67. Cham: Springer, 2017.
"""

import json
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import igraph as ig


def pyomo_preprocess(options=None):
    pass


def pyomo_create_model(options=None, model_options=None) -> pyo.AbstractModel:

    model = pyo.AbstractModel(name="Minimum Volume Problem (LP)")

    model.SetBars = pyo.Set(doc="bars in the ground structure")

    model.SetNodes = pyo.Set(doc="nodes in the ground structure")

    model.SetIncidentBars = pyo.Set(
        model.SetNodes,
        within=model.SetBars,
        doc="indexed set: SetIncidentBars[k] collects bars incident into node k",
    )

    model.node_force_horizontal = pyo.Param(
        model.SetNodes,
        doc="node_force_horizontal[k]: horizontal component of the sum of forces acting on node k. \
            could be external and/or reactions.",
    )

    model.node_force_vertical = pyo.Param(
        model.SetNodes,
        doc="node_force_vertical[k]: vertical component of the sum of forces acting on node k. \
            could be external and/or reactions.",
    )

    model.bar_length = pyo.Param(
        model.SetBars,
        doc="bar_length[k]: length of bar k, constant i.e. assuming zero compliance",
    )

    model.bar_direction_radian = pyo.Param(
        model.SetNodes,
        model.SetBars,
        doc="bar_direction_radians[k,j]: angle between horizontal bar force j in node k, mathematically positive",
    )

    model.tension_stress_limit = pyo.Param(doc="todo")

    model.compression_stress_limit = pyo.Param(doc="todo")

    model.bar_force = pyo.Var(model.SetBars, within=pyo.Reals, doc="todo")

    model.bar_cross_section = pyo.Var(
        model.SetBars, within=pyo.NonNegativeReals, doc="todo"
    )

    @model.Constraint(model.SetNodes)
    def force_equilibrium_horizontal(block, k):
        return (
            sum(
                pyo.cos(block.bar_direction_radian[k, j]) * block.bar_force[j]
                for j in block.SetIncidentBars[k]
            )
            + block.node_force_horizontal[k]
            == 0
        )

    @model.Constraint(model.SetNodes)
    def force_equilibrium_vertical(block, k):
        return (
            sum(
                pyo.sin(block.bar_direction_radian[k, j]) * block.bar_force[j]
                for j in block.SetIncidentBars[k]
            )
            + block.node_force_vertical[k]
            == 0
        )

    @model.Constraint(model.SetBars)
    def tension_limit(block, k):
        return (
            block.bar_cross_section[k] * block.tension_stress_limit - block.bar_force[k]
            >= 0
        )

    @model.Constraint(model.SetBars)
    def compression_limit(block, k):
        return (
            block.bar_cross_section[k] * block.compression_stress_limit
            + block.bar_force[k]
            >= 0
        )

    @model.Objective(sense=pyo.minimize)
    def total_volume(block):
        return pyo.summation(block.bar_length, block.bar_cross_section)

    return model


def _get_bar_length(graph: ig.Graph, layout: ig.Layout, edge: ig.Edge):
    i, j = edge.tuple
    coords_j, coords_i = np.array(layout._coords[j]), np.array(layout._coords[i])
    length = np.linalg.norm(coords_j - coords_i)
    return length


def _get_bar_direction_radian(
    graph: ig.Graph, layout: ig.Layout, vert_index: int, edge_index: int
):

    if vert_index == graph.es[edge_index].tuple[0]:
        k, j = graph.es[edge_index].tuple
    elif vert_index == graph.es[edge_index].tuple[1]:
        j, k = graph.es[edge_index].tuple
    else:
        raise ValueError(
            "edge {} does not appear to be incident into node {}".format(
                edge_index, vert_index
            )
        )

    assert vert_index == k

    coords_k, coords_j = np.array(layout._coords[k]), np.array(layout._coords[j])

    vec_edge = (coords_j - coords_k) / np.linalg.norm(coords_j - coords_k)

    phi = np.arccos(np.dot(vec_edge, [1, 0]))

    if vec_edge[1] < 0:
        phi = 2 * np.pi - phi

    return phi


def pyomo_create_dataportal(model=None, options=None, model_options=None):

    edf = pd.read_csv("./data/bars.csv", index_col="edge ID")
    vdf = pd.read_csv("./data/vertices.csv", index_col="vertex ID")

    g = ig.Graph.DataFrame(edf, vertices=vdf, directed=False, use_vids=True)

    layout = ig.Layout(list(zip(g.vs["coords_x"], g.vs["coords_y"])))

    data_dict = dict()

    data_dict["SetBars"] = list(g.es["label"])
    data_dict["SetNodes"] = list(g.vs["label"])
    data_dict["SetIncidentBars"] = [
        {"index": v["label"], "value": [ind for ind in g.incident(v.index)]}
        for v in g.vs
    ]

    data_dict["node_force_horizontal"] = [
        {"index": v["label"], "value": v["node_force_horizontal"]} for v in g.vs
    ]
    data_dict["node_force_vertical"] = [
        {"index": v["label"], "value": v["node_force_vertical"]} for v in g.vs
    ]

    data_dict["bar_length"] = [
        {"index": e.index, "value": 1000 * _get_bar_length(g, layout, e)} for e in g.es
    ]

    data_dict["bar_direction_radian"] = [
        {
            "index": (g.vs[vert_index]["label"], edge_index),
            "value": _get_bar_direction_radian(g, layout, vert_index, edge_index),
        }
        for vert_index in g.vs.indices
        for edge_index in g.incident(vert_index)
    ]

    data_dict["compression_stress_limit"] = 350
    data_dict["tension_stress_limit"] = 250

    fname = "./data/data.json"
    with open(fname, "w+") as f:
        json.dump(data_dict, f, indent=2)

    data = pyo.DataPortal(model=model)
    data.load(filename=fname)

    return data


def pyomo_print_model(options=None, model=None):
    pass


def pyomo_modify_instance(options=None, model=None, instance=None):
    pass


def pyomo_print_instance(options=None, instance=None):
    pass


def pyomo_save_instance(options=None, instance=None):
    instance.write("mvp.lp")


def pyomo_print_results(options=None, instance=None, results=None):
    # _report_cross_sections(instance=instance)
    # _report_bar_forces(instance=instance)
    # print("")
    pass


# def pyomo_save_results(options=None, instance=None, results=None):
#     # print("call to pyomo_save_results")
#     OUTFILE = Path("results.json")
#     with open(OUTFILE, 'w') as f:
#         sol = dict()
#         for name in instance.component_map(pyo.Var):
#             var = getattr(instance, name)
#             sol[name] = list()
#             for ind in var.keys():

#                 if type(ind)==tuple:
#                     index = [ int(j) if type(j) != str else j for j in index ]
#                 elif type(ind)==str:
#                     index = ind
#                 else:
#                     index = int(ind)

#                 sol[name].append({"index":index, "value":var[index].value})

#         f.write(json.dumps(sol, sort_keys=True, indent=2, separators=(',', ': ')))


def pyomo_postprocess(options=None, instance=None, reults=None):
    pass


def _report_cross_sections(instance=None):
    print("\nCross sections:")
    print("--------------\n")
    print("Bar", "cross_section")
    for index in instance.SetBars:
        print(index, pyo.value(instance.bar_cross_section[index]))


def _report_bar_forces(instance=None):
    print("\nBar forces:")
    print("--------------\n")
    print("Bar", "bar_force")
    for index in instance.SetBars:
        print(index, pyo.value(instance.bar_force[index]))


def _plot_topology(
    vertex_size=5,
    vertex_label_size=12,
    vertex_label_dist=3,
    edge_label_dist=5,
    vertex_label_angle=45 / 180 * np.pi,
    bbox=(400, 400),
    margin=50,
    keep_aspect_ratio=True,
    **kwargs
):

    edf = pd.read_csv("./data/bars.csv", index_col="edge ID")
    vdf = pd.read_csv("./data/vertices.csv", index_col="vertex ID")

    g = ig.Graph.DataFrame(edf, vertices=vdf, directed=False, use_vids=True)

    for v in g.vs:
        if v["node_force_vertical"] and v["node_force_horizontal"]:
            v["color"] = "green"
        elif v["node_force_vertical"] or v["node_force_horizontal"]:
            v["color"] = "yellow"
        else:
            v["color"] = "black"

    out = ig.plot(
        g,
        layout=list(zip(g.vs["coords_x"], g.vs["coords_y"])),
        vertex_size=vertex_size,
        vertex_label_size=vertex_label_size,
        vertex_label_dist=vertex_label_dist,
        edge_label_dist=edge_label_dist,
        vertex_label_angle=vertex_label_angle,
        bbox=bbox,
        margin=margin,
        keep_aspect_ratio=keep_aspect_ratio,
        **kwargs
    )

    return out
