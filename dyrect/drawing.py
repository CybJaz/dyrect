import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_transition_graph(trans_mat, vert_coords, threshold=1.0, node_size=50, edge_size=10, fig=None, ax=None,
                          self_loops=False):
    if fig == None or ax == None:
        fig = plt.figure(figsize=(10, 8))
        # if len(vert_coords[0]) > 2:
        #     ax = fig.add_subplot(projection='3d')
        # else:
        ax = fig.add_subplot()
    dg = nx.DiGraph()

    nnodes = len(trans_mat)

    if self_loops:
        for i in range(nnodes):
            edges = [(i, j, trans_mat[i, j]) for j in range(nnodes) if trans_mat[i, j] > threshold]
            dg.add_weighted_edges_from(edges)
    else:
        for i in range(nnodes):
            edges = [(i, j, trans_mat[i, j]) for j in range(nnodes) if trans_mat[i, j] > threshold and i != j]
            dg.add_weighted_edges_from(edges)

    edge_colors = [e[2] for e in dg.edges.data("weight")]
    cmap = plt.cm.plasma

    span_x = np.max(vert_coords[:, 0]) - np.min(vert_coords[:, 0])
    span_y = np.max(vert_coords[:, 1]) - np.min(vert_coords[:, 1])

    # plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(dg, pos=vert_coords, node_size=node_size)
    nx.draw_networkx_labels(dg, pos=vert_coords[:, :2] + [0.02 * span_x, 0.02 * span_y],
                            labels={i: str(i) for i in range(nnodes)}, font_color='r')
    edges = nx.draw_networkx_edges(dg, pos=vert_coords[:, :2], node_size=node_size,
                                   edge_color=edge_colors, edge_cmap=cmap, width=2, arrowsize=edge_size);

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    # ax = plt.gca()
    ax.set_axis_off()

    return ax


def draw_complex(complex, fig=None, ax=None, circles=False, dim=None, col='blue', alpha=0.4, vlabels=False):
    if fig == None or ax == None:
        fig = plt.figure(figsize=(10, 8))
        print(dim, complex._ambient_dim)
        if dim == 3 or complex._ambient_dim > 2:
            ax = fig.add_subplot(projection='3d')
            ax.set_box_aspect((1.0, 1.0, 0.25))
        else:
            ax = fig.add_subplot()

    if dim == 3 or complex._ambient_dim > 2:
        vertices = np.array([complex.coordinates[v[0]] for v in complex.simplices[0]])
        # print(vertices)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=col, s=30)
        for edge in complex.simplices[1]:
            # print(edge)
            verts = complex.coordinates[list(edge), :]
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], c=col, linewidth=2)

        for tr in complex.simplices[2]:
            verts = complex.coordinates[list(tr), :]
            # print(list(verts))
            t = ax.add_collection3d(Poly3DCollection([verts[:, :3]], color=col, alpha=alpha))

    else:
        vertices = np.array([complex.coordinates[v[0]] for v in complex.simplices[0]])
        ax.scatter(vertices[:, 0], vertices[:, 1], c=col, s=30)

        if vlabels:
            for v in complex.simplices[0]:
                ax.annotate(str(v[0]), (complex.coordinates[v[0], 0], complex.coordinates[v[0], 1]), fontsize=15)

        for edge in complex.simplices[1]:
            verts = complex.coordinates[list(edge), :]
            ax.plot(verts[:, 0], verts[:, 1], c=col, linewidth=2)

        if 2 in complex.simplices:
            for tr in complex.simplices[2]:
                verts = complex.coordinates[list(tr), :]
                tr = plt.Polygon(verts[:, :2], alpha=0.5, color=col)
                plt.gca().add_patch(tr)
            #     t = ax.add_collection3d(Poly3DCollection(verts))

        if circles:
            for (v) in complex.simplices[0]:
                crc = plt.Circle(complex.coordinates[v], complex.epsilon, color='r', alpha=0.1)
                ax.add_patch(crc)

    return fig, ax


def draw_planar_mvf(mvf, mode='crit', fig=None, ax=None, figsize=(10, 8)):
    """
    :param mvf:
    :param mode:
        'critical' - color only critical multivectors
        'all' - color all multivectors
        # TODO: 'morse' - color Morse sets?
    :param fig:
    :param ax:
    :param figsize:
    :return:
    """
    if fig == None or ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    vert_size = 20
    edge_width = 2
    # cmap = plt.cm.get_cmap('gist_rainbow')
    cmap = plt.cm.get_cmap('gist_ncar')

    nv = len(mvf.partition)
    colors = np.array([cmap(c) for c in np.linspace(0, 1., nv)])
    np.random.shuffle(colors)

    if mode == 'all':
        if 2 in mvf.complex.simplices:
            for tr in mvf.complex.simplices[2]:
                verts = mvf.complex.coordinates[list(tr), :]
                tr = plt.Polygon(verts[:, :2], alpha=0.5, color=colors[mvf.simplex2mv(tr)], zorder=0)
                plt.gca().add_patch(tr)

        for edge in mvf.complex.simplices[1]:
            verts = mvf.complex.coordinates[list(edge), :]
            ax.plot(verts[:, 0], verts[:, 1], c=colors[mvf.simplex2mv(edge)], linewidth=edge_width, zorder=5)

        vertices = np.array([mvf.complex.coordinates[v[0]] for v in mvf.complex.simplices[0]])
        vcolors = np.array([colors[mvf.simplex2mv(s)] for s in mvf.complex.simplices[0]])
        for iv, v in enumerate(vertices):
            ax.scatter([vertices[iv, 0]], [vertices[iv, 1]], c=[vcolors[iv]], s=vert_size, zorder=10)
        # ax.scatter(vertices[:, 0], vertices[:, 1], c=vcolors, s=vert_size, zorder=10)

    elif mode == 'crit':
        crit_vert_size = vert_size
        reg_vert_size = crit_vert_size / 2.
        crit_edge_width = edge_width
        reg_edge_width = crit_edge_width / 2.
        shade = 0.3
        if 2 in mvf.complex.simplices:
            for tr in mvf.complex.simplices[2]:
                verts = mvf.complex.coordinates[list(tr), :]
                if mvf.is_critical(mvf.simplex2mv(tr)):
                    tr = plt.Polygon(verts[:, :2], alpha=0.5, color=colors[mvf.simplex2mv(tr)], zorder=0)
                else:
                    tr = plt.Polygon(verts[:, :2], alpha=shade/2., color='k', zorder=0)
                plt.gca().add_patch(tr)

        for edge in mvf.complex.simplices[1]:
            verts = mvf.complex.coordinates[list(edge), :]
            if mvf.is_critical(mvf.simplex2mv(edge)):
                ax.plot(verts[:, 0], verts[:, 1], c=colors[mvf.simplex2mv(edge)], linewidth=crit_edge_width, zorder=5)
            else:
                ax.plot(verts[:, 0], verts[:, 1], c='k', alpha=shade, linewidth=reg_edge_width, zorder=5)

        vertices = np.array([mvf.complex.coordinates[v[0]] for v in mvf.complex.simplices[0]])
        vcolors = np.array([
            colors[mvf.simplex2mv(s)] if mvf.is_critical(mvf.simplex2mv(s)) else np.array([0., 0., 0., shade])
            for s in mvf.complex.simplices[0]])
        vsizes = [crit_vert_size if mvf.is_critical(mvf.simplex2mv(s)) else reg_vert_size
                  for s in mvf.complex.simplices[0]]
        # vcolors = np.array([colors[mvf.simplex2mv(s)] for s in mvf.complex.simplices[0]])
        # print(vspecs)
        ax.scatter(vertices[:, 0], vertices[:, 1], c=vcolors, s=vsizes, zorder=10)

        #     t = ax.add_collection3d(Poly3DCollection(verts))

    return fig, ax


def draw_3D_mvf(mvf, mode='crit', fig=None, ax=None, figsize=(10, 8)):
    """
    :param mvf:
    :param mode:
        'critical' - color only critical multivectors
        'all' - color all multivectors
    :param fig:
    :param ax:
    :param figsize:
    :return:
    """
    if fig == None or ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

    cmap = plt.cm.get_cmap('rainbow')

    nv = len(mvf.partition)
    colors = np.array([cmap(c) for c in np.linspace(0, 1., nv)])
    np.random.shuffle(colors)

    if mode == 'all':
        if 2 in mvf.complex.simplices:
            for tr in mvf.complex.simplices[2]:
                verts = mvf.complex.coordinates[list(tr), :]
                print(verts)
                t = ax.add_collection3d(
                    Poly3DCollection([verts], color=colors[mvf.simplex2mv(tr)], alpha=0.5, zorder=0))
                # tr = plt.Polygon([verts], alpha=0.5, color=colors[mvf.simplex2mv(tr)], zorder=0)
                # plt.gca().add_patch(tr)

        for edge in mvf.complex.simplices[1]:
            verts = mvf.complex.coordinates[list(edge), :]
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], c=colors[mvf.simplex2mv(edge)], linewidth=8, zorder=5)

        vertices = np.array([mvf.complex.coordinates[v[0]] for v in mvf.complex.simplices[0]])
        vcolors = np.array([colors[mvf.simplex2mv(s)] for s in mvf.complex.simplices[0]])
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vcolors, s=60, zorder=10)

    elif mode == 'crit':
        crit_vert_size = 10
        reg_vert_size = crit_vert_size / 2.
        crit_edge_width = 5
        reg_edge_width = crit_edge_width / 2.

        if 2 in mvf.complex.simplices:
            for tr in mvf.complex.simplices[2]:
                verts = mvf.complex.coordinates[list(tr), :]
                if mvf.is_critical(mvf.simplex2mv(tr)):
                    t = ax.add_collection3d(
                        Poly3DCollection([verts], color=colors[mvf.simplex2mv(tr)], alpha=0.5, zorder=0))
                else:
                    t = ax.add_collection3d(Poly3DCollection([verts], color='k', alpha=0.25, zorder=0))

        for edge in mvf.complex.simplices[1]:
            verts = mvf.complex.coordinates[list(edge), :]
            if mvf.is_critical(mvf.simplex2mv(edge)):
                ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], c=colors[mvf.simplex2mv(edge)],
                        linewidth=crit_edge_width, zorder=5)
            # else:
            #     ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], c='k', alpha=0.5, linewidth=reg_edge_width, zorder=5)

        vertices = np.array([mvf.complex.coordinates[v[0]] for v in mvf.complex.simplices[0]])
        vcolors = np.array([
            colors[mvf.simplex2mv(s)] if mvf.is_critical(mvf.simplex2mv(s)) else np.array([0., 0., 0., .5])
            for s in mvf.complex.simplices[0]])
        vsizes = [crit_vert_size if mvf.is_critical(mvf.simplex2mv(s)) else reg_vert_size
                  for s in mvf.complex.simplices[0]]
        # vcolors = np.array([colors[mvf.simplex2mv(s)] for s in mvf.complex.simplices[0]])
        # print(vspecs)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vcolors, s=vsizes, zorder=10)

        #     t = ax.add_collection3d(Poly3DCollection(verts))

    return fig, ax


def draw_poset(poset, fig=None, ax=None, figsize=(10, 8)):
    if fig == None or ax == None:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()
    dg = nx.DiGraph()

    nnodes = poset.npoints
    for i in range(nnodes):
        dg.add_node(i)

    for i in range(nnodes):
        for j in poset.succesors(i):
            dg.add_edge(i, j)

    nx.draw_networkx(dg)
    return ax
