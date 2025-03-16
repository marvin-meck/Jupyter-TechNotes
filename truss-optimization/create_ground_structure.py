import os
import numpy as np
import igraph as ig

if __name__ == "__main__":

    coords = np.array([(i, j) for i in range(2) for j in range(5)])

    g = ig.Graph.Full(n=10)

    g.vs["coords_x"] = coords[:, 1]
    g.vs["coords_y"] = coords[:, 0]

    g.vs["label"] = [chr(97 + i) for i in range(g.vcount())]
    g.es["label"] = [str(e.index) for e in g.es]
    g.es["curved"] = False

    g.vs["node_force_vertical"] = 0
    g.vs["node_force_horizontal"] = 0

    force = 100
    g.vs[0]["node_force_vertical"] = -1 * force
    g.vs[0]["node_force_horizontal"] = -4 * force

    g.vs[5]["node_force_horizontal"] = 4 * force

    g.vs[9]["node_force_vertical"] = 1 * force

    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    _ = g.get_edge_dataframe().to_csv("./data/bars.csv")
    _ = g.get_vertex_dataframe().to_csv("./data/vertices.csv")
