import csv
import os.path

import matplotlib as mpl
mpl.use('macosx')
import numpy as np
from collections import Counter
from dyrect import draw_complex, EpsilonNet, WitnessComplex, PatchedWitnessComplex, \
    all_triangles_intersection_test_2D, all_triangles_intersection_test_3D, is_convex_hull

output_directory = 'experiments_output'

def unit_square_datasize_experiment():
    stats = []
    # set_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # patching_levels = [0, 1, 2, 3, 4]
    set_sizes = [5000, 10000, 15000]
    patching_levels = [0, 1, 2, 3]
    noise = [0.0, 0.1, 0.2]
    output_file = 'unit_square_datasize_experiment.csv'

    seeds = range(25)
    # seeds = [1, 27]

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
        eps=0.05
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


        if not os.path.isfile(output_directory + '/unit_square_datasize_experiment.csv'):
            fieldnames = ["seed", "number of points", "eps", "number of landmarks",
                          "patching_level", "1-Betti number",
                          "nonconvex holes", "intersecting pairs",
                          "3-holes", "4-holes", "5-holes", "6-holes", "7-holes"]
            with open(output_directory + '/' + output_file, 'w', newline='') \
                    as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(fieldnames)
        with open(output_directory + '/' + output_file, 'a', newline='') \
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

unit_square_datasize_experiment()
