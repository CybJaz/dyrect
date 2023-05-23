import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
import numpy as np
from collections import Counter

import gudhi
import dyrect as dy
from dyrect import draw_complex, EpsilonNet, WitnessComplex, PatchedWitnessComplex, \
    all_triangles_intersection_test_2D, all_triangles_intersection_test_3D, is_convex_hull

def draw_example(lms, points, eps, complex):
    fwidth = 20
    fig = plt.figure(figsize=(fwidth, fwidth*0.4))
    rows = 1
    cols = 2

    ambient_dim = points.shape[1]

    if ambient_dim == 2:
        ax = plt.subplot(rows, cols, 1)
        plt.scatter(points[:, 0], points[:, 1], s=0.2)
        plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
        for lm in lms:
            crc = plt.Circle(lm, eps, color='r', alpha=0.05)
            ax.add_patch(crc)
        ax.set_aspect('equal')

        ax = plt.subplot(rows, cols, 2)
    else:
        ax = plt.subplot(rows, cols, 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:,2], s=1.0)
        ax.scatter(lms[:, 0], lms[:, 1], lms[:,2])
        # for lm in lms:
        #     crc = plt.Circle(lm, eps, color='r', alpha=0.05)
        #     ax.add_patch(crc)
        # ax.set_aspect('equal')
        ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
        ax = plt.subplot(rows, cols, 2, projection='3d')
        ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))


    draw_complex(complex, fig=fig, ax=ax, vlabels=True)
    # ax.set_aspect('equal')
    plt.show()


def draw_witnesses(lms, points, witness_complex, barrens=None, labels=True):
    fwidth = 10
    fig = plt.figure(figsize=(fwidth, fwidth))

    ambient_dim = lms.shape[1]

    if ambient_dim == 2:
        ax = plt.subplot()
        draw_complex(witness_complex, fig=fig, ax=ax, vlabels=labels)

        if barrens is None:
            plt.scatter(points[:, 0], points[:, 1], s=3.5)
        else:
            bpoints = np.array([points[x] for x in witness_complex._barren_witnesses[barrens]])
            plt.scatter(bpoints[:, 0], bpoints[:, 1], s=1.5)
        # plt.scatter(lms[:, 0], lms[:, 1], s=20.)
        # plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
        ax.set_aspect('equal')

    # else:
    #     ax = plt.subplot(projection='3d')
    #     draw_complex(complex, fig=fig, ax=ax, vlabels=True)
    #     ax.scatter(points[:, 0], points[:, 1], points[:,2], s=1.0)
    #     ax.scatter(lms[:, 0], lms[:, 1], lms[:,2])
    #     # for lm in lms:
    #     #     crc = plt.Circle(lm, eps, color='r', alpha=0.05)
    #     #     ax.add_patch(crc)
    #     # ax.set_aspect('equal')
    #     ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
    #     ax = plt.subplot(rows, cols, 2, projection='3d')
    #     ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
    # ax.set_aspect('equal')
    plt.show()


def simple_four_hole_example():
    lms = np.array([[0., 0.], [0., .9], [1., 1.], [.8, 0.1]])
    points = np.array([lms[np.mod(i, 4)]/2 + lms[np.mod(i+1, 4)]/2 for i in range(4)] + [[0.5, 0.5]])
    points = np.vstack((lms, points))
    # print(points)

    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=1)
    draw_example(lms, points, 0.8, pwc)
    print(pwc.simplices)
    is_convex_hull(lms)

def nonconvex_six_hole_example():
    lms = np.array([[-1.8, 1.5], [0., 1.], [1., 1.], [1., -0.5], [0., -.3], [-1., -0.5]])
    points = np.array([lms[np.mod(i,6)]/2+lms[np.mod(i+1, 6)]/2 for i in range(7)])
    points = np.vstack((lms, points))
    eps = 0.8

    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=3)
    draw_example(lms, points, eps, pwc)
    print("The hole is convex: " + str(is_convex_hull(lms)))
    all_triangles_intersection_test_2D(pwc)

def simple_five_hole_example():
    lms = np.array([[0., 1.], [1., 1.], [1., -0.5], [0., -1.], [-1., -0.5]])
    points = np.array([lms[np.mod(i,5)]/2+lms[np.mod(i+1, 5)]/2 for i in range(6)])
    points = np.vstack((lms, points))
    # extra_points = np.array([-.5, .25])
    # points = np.vstack((lms, points, extra_points))
    # print(points)

    eps = 0.8
    # wc = WitnessComplex(lms, points, 2, eps)
    # wc = WitnessComplex(lms, points, 2)
    # draw_example(lms, points, eps, wc)
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=2)
    draw_example(lms, points, eps, pwc)

def simple_six_hole_example():
    lms = np.array([[0., 1.], [1., 1.], [1., -0.5], [0., -1.], [-1., -0.5], [-1.5, 0.5]])
    points = np.array([lms[np.mod(i,6)]/2+lms[np.mod(i+1, 6)]/2 for i in range(7)])
    points = np.vstack((lms, points))
    # extra_points = np.array([-.5, .25])
    # points = np.vstack((lms, points, extra_points))
    # print(points)

    eps = 0.8
    # wc = WitnessComplex(lms, points, 2, eps)
    # wc = WitnessComplex(lms, points, 2)
    # draw_example(lms, points, eps, wc)
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=3)
    draw_example(lms, points, eps, pwc)

def flat_six_hole_example():
    lms = np.array([[0., 1.], [1.9, 0.8], [1.9, -0.9], [0., -1.], [-1.88, -0.89], [-1.91, 0.91]])
    points = np.array([lms[np.mod(i,6)]/2+lms[np.mod(i+1, 6)]/2 for i in range(7)])
    points = np.vstack((lms, points))
    print(points)

    eps = 0.8
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=3)
    # pwc = WitnessComplex(lms, points, 2)
    draw_example(lms, points, eps, pwc)

def simple_3d_hole_example():
    lms = np.array([[0., 0.65, 0.], [0.6, 0., 0.],  [0., -.57, 0.], [-0.55, 0., 0.], [0., 0., 1.01], [0., 0., -0.98]])
    points_1simpls = np.array([(lms[x] + lms[y])/2 for (x,y) in
                               [(0, 1), (1, 2), (2, 3), (0, 3),
                                (0, 4), (1, 4), (2, 4), (3, 4),
                                (0, 5), (1, 5), (2, 5), (3, 5)
                                ]])
    points_2simpls = np.array([(lms[x] + lms[y] + lms[z])/3 for (x,y,z) in
                               [(0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 3, 4),
                                (0, 1, 5), (1, 2, 5), (2, 3, 5), (0, 3, 5)
                                ]])
    extra_points = np.array([0,0.5,0])
    points = np.vstack((lms, points_1simpls, points_2simpls, extra_points))

    eps = 0.8
    # wc = WitnessComplex(lms, points, 2, eps)
    # wc = WitnessComplex(lms, points, 2)
    pwc = PatchedWitnessComplex(lms, points, 3, patching_level=3, max_patched_dimensions=3)
    draw_example(lms, points, eps, pwc)

    print(pwc.betti_numbers)


def unit_square_example():
    stats = []
    npoints = 12500

    seeds = range(50)
    seeds = [0]
    for seed in seeds:
        print("Seed: " + str(seed))
        np.random.seed(seed)

        # points = np.random.random((100, 2))
        # eps=0.25

        points = np.random.random((npoints, 2))
        eps=0.025
        # eps=0.18
        EN = EpsilonNet(eps, 0)
        EN.fit(points)
        lms = EN.landmarks
        print("EN done")

        # points = np.random.random((50000, 2))
        # points = np.random.random((20000, 2))
        # eps=0.025

        # gudhi_wc = gudhi.EuclideanStrongWitnessComplex(landmarks=lms, witnesses=points)
        # print("gudhi_wc_done")
        # wc_simplex_tree = gudhi_wc.create_simplex_tree(max_alpha_square=.002)
        # print("st_wc_done")
        # gwc = dy.Complex.from_simplex_tree(wc_simplex_tree, lms)
        # print(lms)
        # print(gwc.simplices)


        # dc = dy.Delaunay2d_complex(lms)
        wc = WitnessComplex(lms, points, 2)
        pwc = PatchedWitnessComplex(lms, points, 2, patching_level=5)
        # draw_witnesses(lms, points, wc, barrens=2)
        # draw_witnesses(lms, points, wc, barrens=2)

        # draw_witnesses(lms, points, wc)
        draw_witnesses(lms, points, pwc, labels=False)
        # draw_witnesses(lms, points, pwc)

        # areax = [0.00, 0.25]
        # areay = [0.15, 0.4]
        # fig, ax = dy.draw_voronoi_cells_2d(lms, order=2, areax=areax, areay=areay,
        #                                    resolution=1000, complex=dc)
        # ax.scatter(points[:, 0], points[:, 1], s=2.1, c='k')
        # draw_example(lms, points, eps, wc)
        # draw_example(lms, points, eps, pwc)

        # print(wc._barren_witnesses)
        # print(wc._weakly_witnessed)
        # print(pwc._patched)

        print(wc.betti_numbers)
        print(pwc.betti_numbers)
        # print(pwc._patched)
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

        stats.append([seed, len(points), eps, len(lms), wc.betti_numbers[1], pwc.betti_numbers[1],
                      len(non_convex_patches), len(intersecting_pairs),
                      patches_sizes[3], patches_sizes[4], patches_sizes[5], patches_sizes[6], patches_sizes[7]])

    # fieldnames = ["seed", "number of points", "eps", "number of landmarks",
    #               "1-Betti before patching", "1-Betti after patching",
    #               "nonconvex holes", "intersecting pairs",
    #               "3-holes", "4-holes", "5-holse", "6-holes", "7-holes"]
    #
    # with open('unit_square_experiment_'+str(npoints)+'.csv', 'w', newline='') as outcsv:
    #     writer = csv.writer(outcsv)
    #     writer.writerow(fieldnames)
    #     # writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
    #     # writer.writeheader()
    #     for row in stats:
    #         writer.writerow(row)

    print("DONE")

def unit_cube_example():
    stats = []
    npoints = 10000

    # seeds = range(50)
    seeds = [0]
    for seed in seeds:
        print("Seed: " + str(seed))
        np.random.seed(seed)

        # points = np.random.random((100, 2))
        # eps=0.25

        points = np.random.random((npoints, 3))
        eps=0.30
        EN = EpsilonNet(eps, 0)
        EN.fit(points)
        lms = EN.landmarks
        print("EN done")

        # points = np.random.random((20000, 2))
        # eps=0.025


        # wc = WitnessComplex(lms, points, 2)
        # pwc2 = PatchedWitnessComplex(lms, points, 2, patching_level=5)
        pwc3 = PatchedWitnessComplex(lms, points, 3, patching_level=5, max_patched_dimensions=3)
        print("Witnesses done")
        # draw_example(lms, points, eps, wc)
        draw_example(lms, points, eps, pwc3)

        # print(wc._barren_witnesses)
        # print(wc._weakly_witnessed)
        # print(pwc._patched)

        # print(wc.betti_numbers)
        print(pwc3.betti_numbers)
        # print(all_triangles_intersection_test_3D(pwc2))

        # # CHECK INTERSECTIONS OF TRIANGLES
        # print(pwc3.betti_numbers)
        intersecting_pairs = all_triangles_intersection_test_3D(pwc3)
        vertices = []
        for tr in intersecting_pairs:
            vertices =  vertices + list(tr)
        vertices = np.unique(vertices)
        for v in vertices:
            print(v, ": ", pwc3.coordinates[v])

        print(intersecting_pairs)
        # triangles = list(set([tr for tr in intersecting_pairs[0]]))
        # dy.draw_triangles_collection(triangles, points, dim=3)

        if len(intersecting_pairs) > 0:
            triangles = list(set([tr for pair in intersecting_pairs for tr in pair]))
            dy.draw_triangles_collection(triangles, points, dim=3)

        # print(pwc._patched)
        # patches_sizes = Counter([len(patch) for patch in pwc._patched[2]])
        # for ps in range(3,8):
        #     if ps not in patches_sizes:
        #         patches_sizes[ps] = 0
        # print(patches_sizes)
        #
        #
        # non_convex_patches = []
        # for patch in pwc._patched[2]:
        #     conv_points = [pwc.coordinates[p] for p in patch]
        #     if not is_convex_hull(conv_points):
        #         non_convex_patches.append(patch)
        # print("Number of non-convex patches: " + str(len(non_convex_patches)))
        # intersecting_pairs = all_triangles_intersection_test_2D(pwc)
        #
        # stats.append([seed, len(points), eps, len(lms), wc.betti_numbers[1], pwc.betti_numbers[1],
        #               len(non_convex_patches), len(intersecting_pairs),
        #               patches_sizes[3], patches_sizes[4], patches_sizes[5], patches_sizes[6], patches_sizes[7]])

    print("DONE")

def torus_example():
    stats = []
    npoints = 10000
    noise = 0.15

    patching_level = 2
    # seeds = range(50)
    seeds = [40]
    for seed in seeds:
        print("Seed: " + str(seed))
        np.random.seed(seed)

        points = dy.torus_sample(npoints, .35)
        random_noise = np.random.random_sample(points.shape) * noise
        points = points + random_noise

        eps=0.25
        EN = EpsilonNet(eps, 0)
        EN.fit(points)
        lms = EN.landmarks
        print("EN done")

        wc = WitnessComplex(lms, points, 2)
        pwc2 = PatchedWitnessComplex(lms, points, 3, patching_level=patching_level)
        # pwc3 = PatchedWitnessComplex(lms, points, 3, patching_level=5, max_patched_dimensions=3)
        draw_example(lms, points, eps, wc)
        draw_example(lms, points, eps, pwc2)

        print(wc.betti_numbers)
        print(pwc2.betti_numbers)
        # print(all_triangles_intersection_test_3D(pwc2))

        patches_sizes = Counter([len(patch) for patch in pwc2._patched[2]])
        for ps in range(3,8):
            if ps not in patches_sizes:
                patches_sizes[ps] = 0
        print(patches_sizes)

        non_convex_patches = []
        intersecting_pairs = all_triangles_intersection_test_3D(pwc2)

        stats.append([seed, len(points), eps, len(lms), patching_level,
                      wc.betti_numbers[1], pwc2.betti_numbers[1],
                      len(non_convex_patches), len(intersecting_pairs),
                      patches_sizes[3], patches_sizes[4], patches_sizes[5], patches_sizes[6], patches_sizes[7]])
    # fieldnames = ["seed", "number of points", "eps", "number of landmarks", "patching level",
    #               "1-Betti before patching", "1-Betti after patching",
    #               "nonconvex holes", "intersecting pairs",
    #               "3-holes", "4-holes", "5-holse", "6-holes", "7-holes"]
    #
    # with open('witness_torus_experiment_'+str(npoints)+'.csv', 'w', newline='') as outcsv:
    #     writer = csv.writer(outcsv)
    #     writer.writerow(fieldnames)
    #     # writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
    #     # writer.writeheader()
    #     for row in stats:
    #         writer.writerow(row)
    print("DONE")

def lorenz_example():
    npoints = 10000

    patching_level = 3
    # seeds = range(50)
    np.random.seed(0)

    points = dy.lorenz_attractor(npoints)
    eps = 1.25
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    print("EN done")

    wc = WitnessComplex(lms, points, 2)
    pwc2 = PatchedWitnessComplex(lms, points, 3, patching_level=patching_level)
    # pwc3 = PatchedWitnessComplex(lms, points, 3, patching_level=5, max_patched_dimensions=3)

    draw_example(lms, points, eps, wc)
    draw_example(lms, points, eps, pwc2)

    print(wc.betti_numbers)
    print(pwc2.betti_numbers)
    # print(all_triangles_intersection_test_3D(pwc2))

    patches_sizes = Counter([len(patch) for patch in pwc2._patched[2]])
    for ps in range(3,8):
        if ps not in patches_sizes:
            patches_sizes[ps] = 0
    print(patches_sizes)

    non_convex_patches = []
    intersecting_pairs = all_triangles_intersection_test_3D(pwc2)


# cgal_2D_intersections_test()
# cgal_2D_convexity_test()

# simple_five_hole_example()
# simple_six_hole_example()
# simple_four_hole_example()
# nonconvex_six_hole_example()
# flat_six_hole_example()
unit_square_example()
# torus_example()

# simple_3d_hole_example()
# unit_cube_example()
# torus_example()

print("END")

# from scipy.spatial.distance import cdist
# from itertools import combinations
# X,Y = np.mgrid[0:1:0.001, 0:1:0.001]
# xy = np.vstack((X.flatten(), Y.flatten())).T
# grid_dists = cdist(xy, lms, 'euclidean')
# points_dists = cdist(points, lms, 'euclidean')
#
# # draw_complex(anc, fig=fig, ax=ax, vlabels=True)
# edge_wareas = dict()
# edge_witnesses = dict()
# verts = [4, 33, 44, 57]
# # verts = [22, 23, 60, 67]
# # limits = [[0.7, 1.], [0.6, 0.9]]
# # verts = [3, 8, 37, 47]
# # limits = [[0, 0.28], [0.4, 0.65]]
# # verts = [5, 25, 51, 54]
# # limits = [[0.18, 0.4], [0.07, 0.3]]
#
# vcoords = np.array([lms[i] for i in verts])
# def get_limits(vertices, offset):
#     xmin, xmax, ymin, ymax = np.min(vcoords[:, 0]), np.max(vcoords[:,0]), np.min(vcoords[:,1]), np.max(vcoords[:, 1])
#     xspan, yspan = xmax-xmin, ymax-ymin
#     return [[xmin - xspan*offset, xmax + xspan*offset], [ymin - yspan*offset, ymax + yspan*offset]]
# limits = get_limits(vcoords, 0.2)
#
# for edge in combinations(verts, 2):
#     edge_wareas[tuple(np.sort(edge))] = []
#     edge_witnesses[tuple(np.sort(edge))] = []
# for x_arg_sorted, x in zip(np.argsort(grid_dists, axis=1), range(len(xy))):
#     wit_edge = tuple(np.sort(x_arg_sorted[:2]))
#     if wit_edge in edge_wareas:
#         edge_wareas[wit_edge].append(x)
# for x_arg_sorted, x in zip(np.argsort(points_dists, axis=1), range(len(points))):
#     wit_edge = tuple(np.sort(x_arg_sorted[:2]))
#     if wit_edge in edge_witnesses:
#         edge_witnesses[wit_edge].append(x)
# # print(edge_witnesses)
#
# fig = plt.figure()
# ax = plt.subplot()
# for points_list in edge_wareas.values():
#     if len(points_list) > 0:
#         e_xy = np.array([xy[i, :] for i in points_list])
#         ax.scatter(e_xy[:, 0], e_xy[:, 1])
# ax.scatter(lms[:, 0], lms[:, 1], color='k', s = 10.5)
# ax.scatter(points[:, 0], points[:, 1], color='k', s = 0.5)
# plt.xlim(limits[0])
# plt.ylim(limits[1])
# plt.show()

# ##### Draw patched witness complex
# fig = plt.figure()
# ax = plt.subplot()
# for points_list in edge_witnesses.values():
#     if len(points_list) > 0:
#         e_xy = np.array([points[i, :] for i in points_list])
#         ax.scatter(e_xy[:, 0], e_xy[:, 1])
# ax.scatter(lms[:, 0], lms[:, 1], color='k', s = 10.5)
# ax.scatter(points[:, 0], points[:, 1], color='k', s = 0.5)
# plt.xlim(limits[0])
# plt.ylim(limits[1])
# plt.show()
