import csv
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
import gudhi

from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist

import dyrect as dy
from dyrect import draw_complex, EpsilonNet, WitnessComplex, VWitnessComplex, PatchedWitnessComplex, \
    all_triangles_intersection_test_2D, all_triangles_intersection_test_3D, is_convex_hull

def make_default_example(npoints = 10000, make_witness=True):
    seed = 0
    print("##########################################################")
    print("Seed: " + str(seed) + "\t Set size: " + str(npoints))
    np.random.seed(seed)
    points = np.random.random((npoints, 3)) - 0.5
    points = points / np.expand_dims(np.linalg.norm(points, axis=1), axis=1)

    eps = 0.15
    EN = EpsilonNet(eps, 0, method='furthest')
    EN.fit(points)
    lms = EN.landmarks
    print("EN done; num of landmarks: ", len(lms))

    wc = None
    if make_witness:
        wc = WitnessComplex(lms, points, 2)
        return points, lms, wc
    else:
        return points, lms

def unit_sphere_example():
    points, lms = make_default_example(30000, make_witness=False)
    timer = dy.Timer()

    dwc = dy.Delaunay3d_complex(lms)
    timer.tock("cgal delaunay")
    gudhi_wc = gudhi.EuclideanWitnessComplex(landmarks=lms, witnesses=points)
    timer.tock("gudhi witness")
    wc = WitnessComplex(lms, points, max_dim=2)
    timer.tock("my witness")
    pwc = PatchedWitnessComplex(lms, points, max_dim=2, max_patched_dimensions=2, patching_level=1)
    timer.tock("patched witness")
    # print(pwc.betti_numbers)
    # print(wc.betti_numbers)
    print(pwc.betti_numbers)
    print(dwc.betti_numbers)

    draw_complex(pwc)
    plt.show()

def clique_complex_experiment():
    points, lms, wc = make_default_example(14000)
    ecwc = dy.EdgeCliqueWitnessComplex(lms, points, 2, max_cliques_dim=100)
    # pwc = PatchedWitnessComplex(lms, points, max_dim=2, max_patched_dimensions=2, patching_level=3)

    print("WC: ", wc.betti_numbers)
    # print("PWC: ", pwc.betti_numbers)
    print("ECWC: ", ecwc.betti_numbers)
    # ecwc.barrens_patching(points, 2, level=3)
    level = 4
    for i in range(1):
        print(i, " patching")
        ecwc.barrens_patching(points, 2, b_param=level)
        # ecwc.voted_barrens_patching(points, 2, level=level)
        print("ECWCp: ", ecwc.betti_numbers)

    # ecwc.voted_barrens_patching(points, 2, level=3)
    # ecwc.barrens_patching(points, 2, level=1)
    # print(ecwc.betti_numbers)
    # ecwc.barrens_patching(points, 2, level=2)
    # print(ecwc.betti_numbers)
    draw_complex(wc)
    fig, ax = draw_complex(ecwc)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
    plt.show()


clique_complex_experiment()
# unit_sphere_example()