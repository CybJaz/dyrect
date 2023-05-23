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

    eps = 0.025
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

unit_sphere_example()