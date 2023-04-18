import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
from itertools import combinations
from dyrect import EpsilonNet, draw_complex, WitnessComplex, Delaunay2d_complex
import dyrect as dy

def draw_voronoi_cells_2d(lms, order=0, areax=[0, 1], areay=[0,1], resolution=10, complex=None):
    fwidth = 10
    # fig = plt.figure(figsize=(fwidth, fwidth*0.4))
    # rows = 1
    # cols = 2

    xpoints = np.linspace(areax[0], areax[1], resolution+1)
    ypoints = np.linspace(areay[0], areay[1], resolution+1)
    X, Y = np.meshgrid(xpoints, ypoints)

    xy = np.vstack((X.flatten(), Y.flatten())).T
    grid_dists = cdist(xy, lms, 'euclidean')
    # points_dists = cdist(points, lms, 'euclidean')
    def get_points_of_interest(verts, dim):
        sim_wareas = dict()
        sim_witnesses = dict()
        for sim in combinations(verts, dim + 1):
            sim_wareas[tuple(np.sort(sim))] = []
            sim_witnesses[tuple(np.sort(sim))] = []
        for x_arg_sorted, x in zip(np.argsort(grid_dists, axis=1), range(len(xy))):
            wit_sim = tuple(np.sort(x_arg_sorted[:dim + 1]))
            if wit_sim in sim_wareas:
                sim_wareas[wit_sim].append(x)
        # for x_arg_sorted, x in zip(np.argsort(points_dists, axis=1), range(len(points))):
        #     wit_sim = tuple(np.sort(x_arg_sorted[:dim + 1]))
        #     if wit_sim in sim_witnesses:
        #         sim_witnesses[wit_sim].append(x)
        return sim_wareas, sim_witnesses

    fig = plt.figure(figsize=(fwidth, fwidth))
    ax = plt.subplot()
    sim_wareas, sim_witnesses = get_points_of_interest(range(len(lms)), order)

    empty_keys = [key for key in sim_wareas.keys() if len(sim_wareas[key]) == 0]
    for key in empty_keys:
            sim_wareas.pop(key)
    if complex is not None:
        not_in_complex = [key for key in sim_wareas.keys() if key not in complex.simplices[order]]
    print(not_in_complex)

    cm = plt.cm.get_cmap('nipy_spectral')(np.linspace(0.05, 1, len(sim_wareas), endpoint=False))
    np.random.seed(0)
    np.random.shuffle(cm)
    # print([(s, len(sim_witnesses[s])) for s in sim_witnesses])
    for isim, sim in enumerate(sim_wareas):
        points_list = sim_wareas[sim]
        if len(points_list) > 0:
            e_xy = np.array([xy[i, :] for i in points_list])
            if complex is not None and sim in not_in_complex:
                area_color = 'k'
                ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, color=area_color)
            else:
                area_color = cm[isim]
                ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, label=sim, color=area_color)
    ax.scatter(lms[:, 0], lms[:, 1], color='k', s=10.5)
    # ax.scatter(points[:, 0], points[:, 1], color='k', s=0.5)
    ax.set_aspect('equal')
    for v in range(len(lms)):
        ax.annotate(str(v), (lms[v, 0], lms[v, 1]), fontsize=5, bbox=dict(boxstyle="round4", fc="w"))
    legend = ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i]._sizes = [30]
    plt.title(str(order) + "-areas of influence")
    # plt.xlim(limits[0])
    # plt.ylim(limits[1])
    # draw_complex(complex, fig=fig, ax=ax)
    return fig, ax

def unit_square_norder_voronoi_cells():
    stats = []
    npoints = 5000

    seeds = [0]
    for seed in seeds:
        print("Seed: " + str(seed))
        np.random.seed(seed)
        points = np.random.random((npoints, 2))
        eps=0.1
        EN = EpsilonNet(eps, 0)
        EN.fit(points)
        lms = EN.landmarks
        print("EN done")
        wc = WitnessComplex(lms, points, 2)
        dc = Delaunay2d_complex(lms)
        # pwc = PatchedWitnessComplex(lms, points, 2, patching_level=4)

        # areax = [0., 1.]
        # areay = [0., 1.]
        areax = [0.56, 0.775]
        areay = [0.21, 0.45]

        fig, ax = draw_voronoi_cells_2d(lms, order=1, areax = areax, areay = areay, resolution=1500, complex=dc)
        draw_complex(wc, fig=fig, ax=ax, alpha=0.2)
        plt.xlim(areax)
        plt.ylim(areay)
        plt.show()

        fig, ax = draw_voronoi_cells_2d(lms, order=2, areax = areax, areay = areay, resolution=1000, complex=dc)
        draw_complex(wc, fig=fig, ax=ax, alpha=0.2)
        plt.xlim(areax)
        plt.ylim(areay)
        plt.show()



unit_square_norder_voronoi_cells()