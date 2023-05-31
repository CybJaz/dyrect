import csv
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
import gudhi
import networkx as nx

from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist

import dyrect as dy
from dyrect import draw_complex, EpsilonNet, WitnessComplex, VWitnessComplex, PatchedWitnessComplex, \
    all_triangles_intersection_test_2D, all_triangles_intersection_test_3D, is_convex_hull

output_directory = 'experiments_output'

def draw_witnesses(lms, points, witness_complex=None, barrens=None, vlabels=True, title=None,
                   fig=None, ax=None):
    if fig is None or ax is None:
        fwidth = 10
        fig = plt.figure(figsize=(fwidth, fwidth))
        ax = plt.subplot()
    ambient_dim = lms.shape[1]

    if ambient_dim == 2:
        if witness_complex is not None:
            draw_complex(witness_complex, fig=fig, ax=ax, vlabels=vlabels, vlabelssize=10)

        if barrens is None:
            ax.scatter(points[:, 0], points[:, 1], s=3.5)
            # plt.scatter(lms[:, 0], lms[:, 1], s=40)
        else:
            bpoints = np.array([points[x] for x in witness_complex._barren_witnesses[barrens]])
            ax.scatter(bpoints[:, 0], bpoints[:, 1], s=1.5)
        ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    # plt.show()

def draw_directed_graph(lms, witness_matrix, vlabels=True, elabels=False, threshold=0):
    fwidth = 10
    fig = plt.figure(figsize=(fwidth, fwidth))
    ambient_dim = lms.shape[1]
    ax = fig.add_subplot()
    max_thick = np.max(witness_matrix)

    n0 = len(lms)
    dp = 0.75
    for i in range(n0):
        for j in range(n0):
            if witness_matrix[i, j] > threshold:
                plt.arrow(lms[i, 0], lms[i, 1],
                          dp * (lms[j, 0] - lms[i, 0]),
                          dp * (lms[j, 1] - lms[i, 1]),
                          width=0.01 * (witness_matrix[i, j]/max_thick), head_width=0.01)
                          # , head_length=0.02)
    plt.scatter(lms[:, 0], lms[:, 1], s=10.)

    if vlabels:
        for v in range(n0):
            ax.annotate(str(v), (lms[v, 0], lms[v, 1]), fontsize=10)
    if elabels:
        for i in range(n0):
            for j in range(i, n0):
                if witness_matrix[i, j] > threshold or witness_matrix[j, i] > threshold:
                    ax.annotate(
                        '[' + str(witness_matrix[i, j]) + ', ' + str(witness_matrix[j, i]) + ']',
                        ((0.3 * lms[i, 0] + 0.7 * lms[j, 0]),
                         (0.3 * lms[i, 1] + 0.7 * lms[j, 1])),
                        fontsize=7)


def make_default_example(npoints=5000, eps=0.1, seed=0, resample=None):
    # npoints = 8500
    # eps = 0.07
    # seed = 15
    # npoints = 5000
    # eps = 0.1
    # seed = 0
    print("##########################################################")
    print("Seed: " + str(seed) + "\t Set size: " + str(npoints))
    np.random.seed(seed)
    points = np.random.random((npoints, 2))
    # points = dy.unit_circle_sample(npoints, noise=0.4)
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    # lms = np.vstack((lms, np.array([[0.1, 0.0]])))
    print("EN done; num of landmarks: ", len(lms))

    if resample is not None:
        np.random.seed(seed+1)
        points = np.random.random((resample, 2))
        # extras = np.random.random((5*resample, 2)) * 0.25 + np.array([0.75, 0.4])
        # print([np.max(extras, axis=0), np.max(extras, axis=1)])
        # print([np.min(extras, axis=0), np.min(extras, axis=1)])
        # points = np.vstack((points, extras))

    wc = WitnessComplex(lms, points, 2)
    return points, lms, wc


def make_torus_example(npoints=5000, eps=0.1, seed=0, resample=None):
    print("##########################################################")
    print("Seed: " + str(seed) + "\t Set size: " + str(npoints))
    np.random.seed(seed)
    points = dy.torus_sample(npoints=npoints)
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    # lms = np.vstack((lms, np.array([[0.1, 0.0]])))
    print("EN done; num of landmarks: ", len(lms))

    if resample is not None:
        np.random.seed(seed+1)
        points = np.random.random((resample, 2))

    wc = WitnessComplex(lms, points, 2)
    return points, lms, wc


def edge_clique_witness_example():
    points, lms, wc = make_default_example(3000, 0.1, 0)

    ac = dy.AlphaComplex(lms)
    ecwc = dy.EdgeCliqueWitnessComplex(lms, points, 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    draw_witnesses(lms, points, wc, vlabels=True, title='witness', fig=fig, ax=ax1)
    draw_witnesses(lms, points, ac, vlabels=True, title='alpha', fig=fig, ax=ax2)
    draw_witnesses(lms, points, ecwc, vlabels=True, title='clique witness', fig=fig, ax=ax3)
    plt.show()

def barrens_cliques():
    points, lms, wc = make_default_example(25000)
    ### przypadek witness nie rowny clique-witness
    # points, lms, wc = make_default_example(8500, 0.07, 15)
    ### przypadek clique-witness nie jest pokompleksem Delauney'a
    # points, lms, wc = make_default_example(3000, 0.1, 0, resample=8000)

    # points, lms, wc = make_default_example(12000, 0.075, 0)
    ac = dy.AlphaComplex(lms)
    ecwc = dy.EdgeCliqueWitnessComplex(lms, points, 2)
    # ecwc.barrens_patching(points, dim=2, level=1)

    draw_directed_graph(lms, ecwc._vmatrix.uni_vmatrix, threshold=0, elabels=False)

    wc_barrens = wc.barren_witnesses(points, 2)
    ecwc_barrens = ecwc.barren_witnesses(points, 2)

    distances = cdist(points, lms, 'euclidean')
    argsort_dists = np.argsort(distances, axis=1)
    distances.sort(axis=1)

    n0 = len(wc.simplices[0])
    witness_matrix = np.zeros((n0, n0))
    uni_witness_matrix = np.zeros((n0, n0))
    points_matrix = np.zeros((n0, n0))
    uni_points_matrix = np.zeros((n0, n0))

    kw = 3
    # for i in wc._barren_witnesses[2]:
    for i in ecwc.barren_witnesses(points, 2, indices=True):
        bsimplex = argsort_dists[i, :(kw+1)]
        for x in range(1, kw+1):
            witness_matrix[bsimplex[0], bsimplex[x]] += 1
    kp = 2
    for i in range(len(points)):
        bsimplex = argsort_dists[i, :(kp+1)]
        for x in range(1, kp+1):
            points_matrix[bsimplex[0], bsimplex[x]] += 1

    # nonzeros = []
    # for i in range(n0):
    #     for j in range(n0):
    #         if witness_matrix[i, j] != 0 and witness_matrix[j, i] != 0:
    #             # v = (witness_matrix[i, j] + witness_matrix[j, i])/2
    #             v = min(witness_matrix[i, j], witness_matrix[j, i])
    #             uni_witness_matrix[i, j] = v
    #             uni_witness_matrix[j, i] = v
    #         if witness_matrix[i, j] > 0:
    #             nonzeros.append(witness_matrix[i,j])
    #         if points_matrix[i, j] != 0 and points_matrix[j, i] != 0:
    #             # v = (witness_matrix[i, j] + witness_matrix[j, i])/2
    #             v = min(points_matrix[i, j], points_matrix[j, i])
    #             # v = max(points_matrix[i, j], points_matrix[j, i])
    #             uni_points_matrix[i, j] = v
    #             uni_points_matrix[j, i] = v
    #         if points_matrix[i, j] > 0:
    #             nonzeros.append(points_matrix[i, j])
    # print(np.mean(nonzeros), np.median(nonzeros))
    #
    # g = nx.from_numpy_matrix(uni_witness_matrix)
    # for clique in nx.find_cliques(g):
    #     if len(clique) > 2:
    #         print(clique)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
    draw_witnesses(lms, wc_barrens, wc, vlabels=True, title='witness', fig=fig, ax=ax1)
    draw_witnesses(lms, points, ac, vlabels=True, title='alpha', fig=fig, ax=ax2)
    draw_witnesses(lms, ecwc_barrens, ecwc, vlabels=True, title='clique witness', fig=fig, ax=ax3)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    # ecwc.barrens_patching(points, dim=2, level=3)
    # draw_witnesses(lms, ecwc_barrens, ecwc, vlabels=False, title='patched clique witness', fig=fig, ax=ax1)
    # ecwc2 = dy.EdgeCliqueWitnessComplex(lms, points, 2, max_cliques_dim=2)
    # ecwc2.voted_barrens_patching(points, dim=2, level=5)
    # # ecwc2.voted_barrens_patching(points, dim=2, level=3)
    # draw_witnesses(lms, ecwc_barrens, ecwc2, vlabels=False, title='voted patched clique witness', fig=fig, ax=ax2)
    # dy.draw_barren_witnesses(ecwc, points=points, order=2)

    # draw_complex(wc, vlabels=True)
    # draw_witnesses(lms, points, ac, vlabels=True)

    # draw_witnesses(lms, points, wc, barrens=2, vlabels=True)

    areax = [0.42, 0.62]
    areay = [0.27, 0.47]
    # areax = [0.32, 0.52]
    # areay = [0.25, 0.45]
    res = 512
    # fig, ax = dy.draw_directed_voronoi_cells_2d(lms, order=2, complex=None, resolution=res,
    #                                             areax=areax, areay=areay)
    # draw_witnesses(lms, points=points, ax=ax, fig=fig)
    # # dy.draw_barren_witnesses(wc, points=points, ax=ax, fig=fig, order=2)
    # fig, ax = dy.draw_voronoi_cells_2d(lms, order=2, complex=None, resolution=res,
    #                                    areax=areax, areay=areay)
    # dy.draw_barren_witnesses(wc, points=points, ax=ax, fig=fig, order=2)
    # fig, ax = dy.draw_directed_voronoi_cells_2d(lms, order=3, complex=None, resolution=res,
    #                                             areax=areax, areay=areay)
    # draw_witnesses(lms, points=points, ax=ax, fig=fig)

    # dy.draw_barren_witnesses(wc, points=points, ax=ax, fig=fig, order=2)
    # fig, ax = dy.draw_voronoi_cells_2d(lms, order=3, complex=None, resolution=res,
    #                                    areax=areax, areay=areay)
    # dy.draw_barren_witnesses(wc, points=points, ax=ax, fig=fig, order=2)

    # draw_directed_graph(lms, uni_points_matrix, threshold=0, elabels=True)
    # draw_directed_graph(lms, witness_matrix, threshold=0, elabels=True)
    # draw_directed_graph(lms, uni_witness_matrix, threshold=0, elabels=True)

    # print(ecwc.betti_numbers)
    # print(ecwc2.betti_numbers)
    # print(wc.betti_numbers)
    plt.show()


barrens_cliques()
# edge_clique_witness_example()

