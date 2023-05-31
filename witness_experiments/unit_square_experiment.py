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

output_directory = 'experiments_output'

def draw_witnesses(lms, points, witness_complex, barrens=None, vlabels=True):
    fwidth = 10
    fig = plt.figure(figsize=(fwidth, fwidth))
    ambient_dim = lms.shape[1]

    if ambient_dim == 2:
        ax = plt.subplot()
        draw_complex(witness_complex, fig=fig, ax=ax, vlabels=vlabels)

        if barrens is None:
            plt.scatter(points[:, 0], points[:, 1], s=3.5)
            # plt.scatter(lms[:, 0], lms[:, 1], s=40)
        else:
            bpoints = np.array([points[x] for x in witness_complex._barren_witnesses[barrens]])
            plt.scatter(bpoints[:, 0], bpoints[:, 1], s=1.5)
        ax.set_aspect('equal')
    plt.show()

def make_default_example():
    npoints = 5000
    seed = 0
    print("##########################################################")
    print("Seed: " + str(seed) + "\t Set size: " + str(npoints))
    np.random.seed(seed)
    points = np.random.random((npoints, 2))
    eps = 0.1
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    # lms = np.vstack((lms, np.array([[0.1, 0.0]])))
    print("EN done; num of landmarks: ", len(lms))

    wc = WitnessComplex(lms, points, 2)
    return points, lms, wc

def unit_square_intro_example():
    npoints = 5000
    # patching_levels = [0, 1, 2]

    seed = 0

    print("##########################################################")
    print("Seed: " + str(seed) + "\t Set size: " + str(npoints))
    np.random.seed(seed)

    points = np.random.random((npoints, 2))
    eps = 0.1
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    print("EN done; num of landmarks: ", len(lms))

    # np.random.seed(seed)
    # points = np.random.random((10*npoints, 2))

    wc = WitnessComplex(lms, points, 2, alpha=0.0, beta_sq=0.0)
    # wc = WitnessComplex(lms, points, 2, alpha=0.0, beta_sq=0.002)
    draw_witnesses(lms, points, wc, vlabels=True)

    vwc = VWitnessComplex(lms, points, 2, alpha=0.0, v=[0,1,0])
    print([len(vwc.simplices[d]) for d in vwc.simplices])
    draw_witnesses(lms, points, vwc, vlabels=True)

    # gudhi_wc = gudhi.EuclideanWitnessComplex(landmarks=lms, witnesses=points)
    # print("gudhi_wc_done")
    # wc_simplex_tree = gudhi_wc.create_simplex_tree(max_alpha_square=.002)
    # print("st_wc_done")
    # gwc = dy.Complex.from_simplex_tree(wc_simplex_tree, lms)
    # print(lms)
    # print(gwc.simplices)

    # draw_witnesses(lms, points, gwc, vlabels=True)
    print(wc.betti_numbers)


def unit_square_barren_witnesses_example():
    points, lms, wc = make_default_example()

    alpha_complex = gudhi.AlphaComplex(points=lms)
    stree = alpha_complex.create_simplex_tree()
    simplices = {0:[], 1:[], 2:[]}
    for (s, _) in stree.get_simplices():
        simplices[len(s)-1].append(tuple(np.sort(s)))
    ac = dy.Complex(simplices=simplices, coords=lms)
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=3)
    # draw_witnesses(lms, points, wc, vlabels=True)

    ambient_dim = lms.shape[1]

    distances = cdist(points, lms, 'euclidean')
    argsort_dists = np.argsort(distances, axis=1)
    distances.sort(axis=1)

    k = 2
    innately_barren = []
    incidentally_barren = []
    for i in wc._barren_witnesses[k]:
        bsimplex = tuple(np.sort(argsort_dists[i, :(k+1)]))
        if bsimplex in ac.simplices[k]:
            incidentally_barren.append(points[i])
        else:
            innately_barren.append(points[i])
    innately_barren = np.array(innately_barren)
    incidentally_barren = np.array(incidentally_barren)

    # draw_complex(ac)
    # draw_complex(wc)
    # draw_complex(pwc)

    k = k + 1
    fwidth = 10
    fig = plt.figure(figsize=(fwidth, fwidth))
    ax = plt.subplot()
    dy.draw_voronoi_cells_2d(lms, fig=fig, ax=ax, order=k, resolution=1000)
                             # labels=True,  areax=[0.55,0.8], areay=[0.2, 0.48])
    draw_complex(wc, fig=fig, ax=ax)

    plt.scatter(innately_barren[:, 0], innately_barren[:, 1], s=3.5)
    plt.scatter(incidentally_barren[:, 0], incidentally_barren[:, 1], s=3.5)

    # if barrens is None:
    #     plt.scatter(points[:, 0], points[:, 1], s=3.5)
    #     # plt.scatter(lms[:, 0], lms[:, 1], s=40)
    # else:
    #     bpoints = np.array([points[x] for x in witness_complex._barren_witnesses[barrens]])
    #     plt.scatter(bpoints[:, 0], bpoints[:, 1], s=1.5)
    ax.set_aspect('equal')
    plt.show()

def unit_square_datasize_experiment():
    stats = []
    filename = 'unit_square_datasize_experiment'
    # set_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # patching_levels = [0, 1, 2, 3, 4]
    set_sizes = [15000, 20000, 25000, 30000, 35000, 40000, 50000]
    patching_levels = [0, 1, 2]
    eps = 0.05

    seeds = range(25)
    # seeds = [1, 27]


    # filename = 'small_unit_square_datasize_experiment'
    # # set_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
    # set_sizes = [27500, 30000, 35000, 40000, 45000, 50000]
    # patching_levels = [0]
    # eps = 0.1

    parameters = []
    for npoints in set_sizes:
        for seed in seeds:
            for pl in patching_levels:
                parameters.append((seed, npoints, pl))

    for seed, npoints, patching_level in parameters:
        print("##########################################################")
        print("Seed: " + str(seed) + "\t Set size: " + str(npoints) +
              "\t Patching level: " + str(patching_level))
        np.random.seed(seed)

        points = np.random.random((npoints, 2))
        EN = EpsilonNet(eps, 0)
        EN.fit(points)
        lms = EN.landmarks
        print("EN done")

        # if patching_level == 0:
        #     pwc = WitnessComplex(lms, points, 2)
        # else:
        pwc = PatchedWitnessComplex(lms, points, 2, patching_level=patching_level)

        print(pwc.betti_numbers)

        if 2 not in pwc._patched:
            pwc._patched[2] = []

        patches_sizes = Counter([len(patch) for patch in pwc._patched[2]])
        for ps in range(3, 8):
            if ps not in patches_sizes:
                patches_sizes[ps] = 0
        print(patches_sizes)

        non_convex_patches = []
        for patch in pwc._patched[2]:
            conv_points = [pwc.coordinates[p] for p in patch]
            if not is_convex_hull(conv_points):
                non_convex_patches.append(patch)
        print("Number of non-convex patches: " + str(len(non_convex_patches)))
        print("Non-convex patches: " + str(non_convex_patches))
        intersecting_pairs = all_triangles_intersection_test_2D(pwc)

        stats.append([seed, len(points), eps, len(lms),
                      patching_level, pwc.betti_numbers[1],
                      len(non_convex_patches), len(intersecting_pairs),
                      patches_sizes[3], patches_sizes[4], patches_sizes[5], patches_sizes[6], patches_sizes[7]])


        if not os.path.isfile(output_directory + '/' + filename + '.csv'):
            fieldnames = ["seed", "number of points", "eps", "number of landmarks",
                          "patching_level", "1-Betti number",
                          "nonconvex holes", "intersecting pairs",
                          "3-holes", "4-holes", "5-holes", "6-holes", "7-holes"]
            with open(output_directory + '/' + filename + '.csv', 'w', newline='') \
                    as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(fieldnames)
        with open(output_directory + '/' + filename + '.csv', 'a', newline='') \
                as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(stats[-1])

    #     for row in stats:
    #         writer.writerow(row)
    # fieldnames = ["seed", "number of points", "eps", "number of landmarks",
    #               "patching_level", "1-Betti number",
    #               "nonconvex holes", "intersecting pairs",
    #               "3-holes", "4-holes", "5-holes", "6-holes", "7-holes"]

    # with open(output_directory + '/unit_square_datasize_experiment.csv', 'w', newline='')\
    #         as outcsv:
    #     writer = csv.writer(outcsv)
    #     writer.writerow(fieldnames)
    #     for row in stats:
    #         writer.writerow(row)

    print("DONE")

# unit_square_intro_example()
# unit_square_datasize_experiment()
unit_square_barren_witnesses_example()