import matplotlib.pyplot as plt
import numpy as np
# import scipy as sc
import dyrect as dy
from dyrect import draw_complex, EpsilonNet, WitnessComplex, PatchedWitnessComplex

def draw_example(lms, points, eps, complex):
    fwidth = 20
    fig = plt.figure(figsize=(fwidth, fwidth*0.4))
    rows = 1
    cols = 2

    ax = plt.subplot(rows, cols, 1)
    plt.scatter(points[:, 0], points[:, 1], s=0.2)
    plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    for lm in lms:
        crc = plt.Circle(lm, eps, color='r', alpha=0.05)
        ax.add_patch(crc)
    ax.set_aspect('equal')

    ax = plt.subplot(rows, cols, 2)
    draw_complex(complex, fig=fig, ax=ax, vlabels=True)
    ax.set_aspect('equal')
    plt.show()

def simple_four_hole_example():
    lms = np.array([[0., 0.], [0., .9], [1., 1.], [.8, 0.1]])
    points = np.array([lms[np.mod(i, 4)]/2 + lms[np.mod(i+1, 4)]/2 for i in range(4)])
    points = np.vstack((lms, points))
    # print(points)

    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=1)
    draw_example(lms, points, 0.8, pwc)

def simple_six_hole_example():
    lms = np.array([[0., 1.], [1., 1.], [1., -0.5], [0., -1.], [-1., -0.5], [-1.5, 0.5]])
    points = np.array([lms[np.mod(i,6)]/2+lms[np.mod(i+1, 6)]/2 for i in range(7)])
    points = np.vstack((lms, points))
    # print(points)

    eps = 0.8
    # wc = WitnessComplex(lms, points, 2, eps)
    # wc = WitnessComplex(lms, points, 2)
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=1)
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

def unit_square_example():
    np.random.seed(1)
    # crc1 = unit_circle_sample(4000, 0.75) + [1.1,0]
    # crc2 = unit_circle_sample(4000, 0.75) - [1.1,0]
    # points = np.append(crc1, crc2, axis=0)
    # eps=.25

    points = np.random.random((5000, 2))
    eps=0.05

    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    print("EN done")

    wc = WitnessComplex(lms, points, 2)
    draw_example(lms, points, eps, wc)
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=3)
    draw_example(lms, points, eps, pwc)

# simple_six_hole_example()
simple_four_hole_example()
# flat_six_hole_example()
# unit_square_example()

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
