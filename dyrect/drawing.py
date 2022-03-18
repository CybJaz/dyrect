import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_transition_graph(trans_mat, vert_coords, threshold=1.0, node_size=50, edge_size=10):
    dg = nx.DiGraph()

    nnodes = len(trans_mat)

    for i in range(nnodes):
        edges = [(i, j, trans_mat[i, j]) for j in range(nnodes) if trans_mat[i, j] > threshold]
        dg.add_weighted_edges_from(edges)

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

def draw_complex(complex, fig=None, ax=None, circles=True, col='blue', alpha=0.4):
    if fig == None or ax == None:
        fig = plt.figure(figsize=(10, 8))
        if complex.dimension > 2:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()

    if complex.dimension > 2:
        vertices = np.array([complex.coordinates[v[0]] for v in complex.simplices[0]])
        # print(vertices)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=col, s=30)
        for edge in complex.simplices[1]:
            print(edge)
            verts = complex.coordinates[list(edge), :]
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], c=col, linewidth=2)

        for tr in complex.simplices[2]:
            verts = complex.coordinates[list(tr), :]
            t = ax.add_collection3d(Poly3DCollection(verts, color=col, alpha=alpha))

    else:
        vertices = np.array([complex.coordinates[v[0]] for v in complex.simplices[0]])
        ax.scatter(vertices[:, 0], vertices[:, 1], c=col, s=30)
        for edge in complex.simplices[1]:
            verts = complex.coordinates[list(edge), :]
            ax.plot(verts[:, 0], verts[:, 1], c=col, linewidth=2)

        for tr in complex.simplices[2]:
            verts = complex.coordinates[list(tr), :]
            tr = plt.Polygon(verts, alpha=0.5, color=col)
            plt.gca().add_patch(tr)
        #     t = ax.add_collection3d(Poly3DCollection(verts))

        if circles:
            for (v) in complex.simplices[0]:
                crc = plt.Circle(complex.coordinates[v], complex.epsilon, color='r', alpha=0.1)
                ax.add_patch(crc)

    return fig, ax
