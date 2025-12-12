"""Branching network optimization problem as described by Samir (1979) [1, Section 3]. 

Finds the optimal piping configuration that minimizes total invest, where the diameters of the pipes connecting two nodes are 
chosen from a discrete set. 
The desicion variables correspond to the length percentage of the overall pipe lenght that is chosen for a given diameter. 
Branching networks, i.e. acyclic graphs, are considered in this model only, such that the flows in all links of the graph are
known a priori and thus the associated head loss per unit length and for a given diameter can be computed in advance (provided as 
parameter). 
In the base model, no pumping equipment or valves are considered and invest is assumed linear with different cost per unit length
per diameter. 

The model is implemented using the Pyomo modeling language [2]. 

Author: Marvin Meck 

References:
-----------

    [1] Shamir, Uri (1979): Optimization in water distribution systems engineering. In M. Avriel, R. S. Dembo (Eds.): Engineering
        Optimization, vol. 11. Berlin, Heidelberg: Springer Berlin Heidelberg (Mathematical Programming Studies), pp. 65–84.

    [2] Hart, William E., William E. Hart; Watson, Jean-Paul; Laird, Carl D.; Nicholson, Bethany L.; Siirola, John D. (2017): Pyomo. 
        Optimization Modeling in Python. Second edition /  William E Hart [and six others]. Cham: Springer (Springer Optimization and Its Applications, 67).
"""

import pyomo.environ as pyo


def pyomo_create_model(**kwargs):
    model = pyo.AbstractModel()

    model.IndexNodes = pyo.Set()
    model.IndexLinks = pyo.Set(domain=model.IndexNodes*model.IndexNodes)
    model.IndexDiameters = pyo.Set()

    model.length_link = pyo.Param(model.IndexLinks)
    model.diameter = pyo.Param(model.IndexDiameters)
    model.head_losses = pyo.Param(model.IndexLinks*model.IndexDiameters)
    model.cost_per_unit_length = pyo.Param(model.IndexDiameters)
    model.head_lb = pyo.Param(model.IndexNodes)
    model.head_ub = pyo.Param(model.IndexNodes)

    model.fraction_segment = pyo.Var(
        model.IndexLinks*model.IndexDiameters,
        domain=pyo.NonNegativeReals,
        bounds=(0,1)
    )

    model.head = pyo.Var(
        model.IndexNodes,
        domain=pyo.NonNegativeReals
    )


    @model.Expression(model.IndexLinks)
    def invest_per_link(self,i,j):
        return sum(self.cost_per_unit_length[k]*self.fraction_segment[i,j,k] for k in self.IndexDiameters)


    @model.Expression(model.IndexLinks)
    def head_losses_per_link(self,i,j):
        return sum(self.head_losses[i,j,k] * self.fraction_segment[i,j,k] for k in self.IndexDiameters)


    @model.Constraint(model.IndexLinks)
    def fraction_segments_rule(self,i,j):
        return sum(self.fraction_segment[i,j,:]) == 1


    @model.Constraint(model.IndexLinks)
    def bernoulli_rule(self,i,j):
        return self.head[j] + self.head_losses_per_link[i,j] == self.head[i]

    @model.Constraint(model.IndexNodes)
    def min_head_rule(self,k):
        return self.head_lb[k],self.head[k],self.head_ub[k]
    
    @model.Objective(sense=pyo.minimize)
    def invest(self):
        return pyo.summation(self.invest_per_link,self.length_link, index=self.IndexLinks)

    return model
