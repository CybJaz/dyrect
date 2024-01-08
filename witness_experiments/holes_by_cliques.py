import csv
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')

from itertools import permutations
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
            ax.scatter(points[:, 0], points[:, 1], s=2.25, c='g', alpha=0.5)
            # plt.scatter(lms[:, 0], lms[:, 1], s=40)
        else:
            bpoints = np.array([points[x] for x in witness_complex._barren_witnesses[barrens]])
            ax.scatter(bpoints[:, 0], bpoints[:, 1], s=1.5)
        ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    # plt.show()

def draw_directed_graph(lms, witness_matrix, vlabels=True, elabels=False, threshold=0, title=None):
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
    if title is not None:
        ax.set_title(title)

def make_ring_example(npoints=5000, eps=0.1, seed=0, resample=None, noise=0.01):
    print("##########################################################")
    print("Seed: " + str(seed) + "\t Set size: " + str(npoints))
    np.random.seed(seed)
    points = dy.unit_circle_sample(npoints, noise=noise)
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    print("EN done; num of landmarks: ", len(lms))

    if resample is not None:
        np.random.seed(seed+1)
        points = dy.unit_circle_sample(resample, noise=noise)

    wc = WitnessComplex(lms, points, 2)
    return points, lms, wc

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
    # points = dy.unit_circle_sample(npoints, noise=0.8)
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
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
    points, lms, wc = make_default_example(2000, 0.1, 0, resample=8000)

    ac = dy.AlphaComplex(lms)
    ecwc = dy.EdgeCliqueWitnessComplex(lms, points, 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
    draw_witnesses(lms, points, wc, vlabels=True, title='witness BN:' + str(wc.betti_numbers), fig=fig, ax=ax1)
    draw_witnesses(lms, points, ac, vlabels=True, title='alpha BN:' + str(ac.betti_numbers), fig=fig, ax=ax2)
    draw_witnesses(lms, points, ecwc, vlabels=True, title='clique BN:' + str(ecwc.betti_numbers), fig=fig, ax=ax3)

    draw_directed_graph(lms, ecwc._vmatrices[1].matrix, threshold=0, elabels=False, title="1-edge witness graph")
    draw_directed_graph(lms, ecwc._vmatrices[2].matrix, threshold=0, elabels=False, title="2-edge witness graph")


    ### (2000, 0.1, 0, resample=8000)
    res = 2**11
    voi = [5, 16, 42, 50]
    # simplices_of_interest = [[5, 42, 50], [5, 16, 42]]
    simplices_of_interest = [[5, 16, 50], [16, 42, 50]]

    fig, ax = dy.draw_directed_voronoi_cells_2d(lms, order=3,
                                                voi=voi, soi=simplices_of_interest,
                                                resolution=res, scale=0.95)
    draw_witnesses(lms, points=points, ax=ax, fig=fig)

    plt.show()

def witness_fail_statistics_sample_size_statistics():

    wc_missing_triangles_ratio = dict()
    pwc_missing_triangles_ratio = dict()
    pwc_extra_triangles_ratio = dict()
    ac_triangles_count = dict()

    # 1st betti numbers
    wc_bn = dict()
    pwc_bn = dict()

    # parameter types
    parameter_types = ['sample_size', 'epsilon']
    parameter_type = 0

    if parameter_type == 0:
        parameter_name = "sample size"
        parameter_values = [5000, 7500, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
    elif parameter_type == 1:
        parameter_name = "epsilon"
        parameter_values = [0.05, 0.075, 0.1, 0.15, 0.2]

    for pv in parameter_values:
        wc_missing_triangles_ratio[pv] = []
        pwc_missing_triangles_ratio[pv] = []
        pwc_extra_triangles_ratio[pv] = []
        ac_triangles_count[pv] = []

        wc_bn[pv] = []
        pwc_bn[pv] = []

        for seed in range(15):
            if parameter_type == 0:
                points, lms, wc = make_default_example(2000, 0.1, seed, resample=pv)
            elif parameter_type == 1:
                points, lms, wc = make_default_example(5000, pv, seed, resample=10000)
            ac = dy.AlphaComplex(lms)
            pwc = dy.EdgeCliqueWitnessComplex(lms, points, 2)

            wc_bn[pv].append(wc.betti_numbers[1])
            pwc_bn[pv].append(pwc.betti_numbers[1])

            ac_triangles = set(ac.simplices[2])
            wc_triangles = set(wc.simplices[2])
            pwc_triangles = set(pwc.simplices[2])

            wc_missing_triangles = ac_triangles.difference(wc_triangles)
            pwc_missing_triangles = ac_triangles.difference(pwc_triangles)
            pwc_extra_triangles = pwc_triangles.difference(ac_triangles)

            wc_missing_triangles_ratio[pv].append(len(wc_missing_triangles)/len(ac_triangles))
            pwc_missing_triangles_ratio[pv].append(len(pwc_missing_triangles)/len(ac_triangles))
            pwc_extra_triangles_ratio[pv].append(len(pwc_extra_triangles)/len(ac_triangles))
            ac_triangles_count[pv].append(len(ac_triangles))

    avg_wc_missing = [np.mean(wc_missing_triangles_ratio[ss]) for ss in parameter_values]
    avg_pwc_missing = [np.mean(pwc_missing_triangles_ratio[ss]) for ss in parameter_values]
    avg_pwc_extra = [np.mean(pwc_extra_triangles_ratio[ss]) for ss in parameter_values]
    std_wc_missing = [np.std(wc_missing_triangles_ratio[ss]) for ss in parameter_values]
    std_pwc_missing = [np.std(pwc_missing_triangles_ratio[ss]) for ss in parameter_values]
    std_pwc_extra = [np.std(pwc_extra_triangles_ratio[ss]) for ss in parameter_values]

    plt.figure()
    plt.errorbar(parameter_values, avg_wc_missing, std_wc_missing, label="wc missing triangles ratio")
    plt.errorbar(parameter_values, avg_pwc_missing, std_pwc_missing, label="pwc missing triangles ratio")
    plt.errorbar(parameter_values, avg_pwc_extra, std_pwc_extra, label="pwc extra triangles ratio")
    plt.legend()
    plt.xlabel(parameter_name)

    avg_wc_bn = [np.mean(wc_bn[s]) for s in parameter_values]
    std_wc_bn = [np.std(wc_bn[s]) for s in parameter_values]
    avg_pwc_bn = [np.mean(pwc_bn[s]) for s in parameter_values]
    std_pwc_bn = [np.std(pwc_bn[s]) for s in parameter_values]

    plt.figure()
    plt.errorbar(parameter_values, avg_wc_bn, std_wc_bn, label="wc 1-betti number")
    plt.errorbar(parameter_values, avg_pwc_bn, std_pwc_bn, label="pwc 1-betti number")
    plt.legend()
    plt.xlabel(parameter_name)

    plt.show()

def barren_lands():
    points, lms, wc = make_default_example(2000, 0.1, 0, resample=8000)
    ac = dy.AlphaComplex(lms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    draw_witnesses(lms, points, wc, vlabels=True, title='witness BN:' + str(wc.betti_numbers), fig=fig, ax=ax1)
    draw_witnesses(lms, points, ac, vlabels=True, title='alpha BN:' + str(ac.betti_numbers), fig=fig, ax=ax2)
    fig.savefig('witness_alpha.pdf')

    fig,ax = dy.draw_barren_lands(wc, ac, witnesses=points)
    fig.savefig('barren_lands.pdf')

    plt.show()

def witness_complex_patching_example():
    points, lms, wc = make_default_example(2000, 0.1, 2, resample=6000)
    # points, lms, wc = make_ring_example(2000, 0.25, 0, noise=1.0, resample=7000)

    b_param = 3

    ac = dy.AlphaComplex(lms)
    pwc = WitnessComplex.make_a_copy(wc)
    egraph = pwc.barrens_patching(points, 2, b_param)
    # ecwc = dy.EdgeCliqueWitnessComplex(lms, points, 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
    draw_witnesses(lms, points, wc, vlabels=True, title='witness', fig=fig, ax=ax1)
    draw_witnesses(lms, points, ac, vlabels=True, title='delaunay', fig=fig, ax=ax2)
    draw_witnesses(lms, points, pwc, vlabels=True, title='(2,2)-patched', fig=fig, ax=ax3)

    fig, ([[ax1, ax2, ax3], [ax4, ax5, ax6]]) = plt.subplots(2, 3, figsize=(15, 10))
    draw_witnesses(lms, points, wc, vlabels=True, title='witness BN:' + str(wc.betti_numbers), fig=fig, ax=ax1)
    draw_witnesses(lms, points, ac, vlabels=True, title='alpha BN:' + str(ac.betti_numbers), fig=fig, ax=ax2)
    draw_witnesses(lms, points, pwc, vlabels=True, title='patched BN:' + str(pwc.betti_numbers), fig=fig, ax=ax3)

    barren_witnesses = wc.get_barren_witnesses(points, 2)
    draw_witnesses(lms, barren_witnesses, wc, vlabels=True, title='barrens 2', fig=fig, ax=ax4)
    barren_witnesses_2 = pwc.get_barren_witnesses(points, 2)
    draw_witnesses(lms, barren_witnesses_2, pwc, vlabels=True, title='barrens 2', fig=fig, ax=ax6)

    egraph2 = pwc.get_edge_witness_graph(barren_witnesses_2, b_param)

    # draw_directed_graph(lms, egraph.matrix, threshold=0, elabels=False, title="1-edge witness graph")
    draw_directed_graph(lms, egraph.uni_vmatrix, threshold=0, elabels=False, title="1-edge witness graph before patching")
    draw_directed_graph(lms, egraph2.uni_vmatrix, threshold=0, elabels=False, title="1-edge witness graph after patching")

    egraph2 = pwc.barrens_patching(points, 2, b_param+1)
    draw_witnesses(lms, points, pwc, vlabels=True, title='patched x2 BN:' + str(pwc.betti_numbers), fig=fig, ax=ax5)
    draw_directed_graph(lms, egraph2.uni_vmatrix, threshold=0, elabels=False, title="1-edge witness graph before the second patching")

    plt.show()


# edge_clique_witness_example()
# witness_complex_patching_example()

# barren_lands()

witness_fail_statistics_sample_size_statistics()