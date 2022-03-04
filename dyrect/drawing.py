import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl


def draw_transition_graph(trans_mat, vert_coords, threshold=1.0, node_size=50, edge_size=10):
    dg = nx.DiGraph()

    # threshold = 0.15
    nnodes = len(trans_mat)

    for i in range(nnodes):
        edges = [(i, j, trans_mat[i, j]) for j in range(nnodes) if trans_mat[i, j] > threshold]
        dg.add_weighted_edges_from(edges)
    # for i in range(nnodes):
    #     if prob_matrix[i,i] > threshold:
    #         print([i,prob_matrix[i,i]])

    edge_colors = [e[2] for e in dg.edges.data("weight")]
    cmap = plt.cm.plasma

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(dg, pos=vert_coords, node_size=node_size)
    edges = nx.draw_networkx_edges(dg, pos=vert_coords[:,:2], node_size=node_size,
                        edge_color=edge_colors, edge_cmap=cmap, width=2, arrowsize=edge_size);

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    ax = plt.gca()
    ax.set_axis_off()

    return ax