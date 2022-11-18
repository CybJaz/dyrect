import matplotlib.pyplot as plt
# import miniball
import numpy as np
# import scipy as sc
import dyrect as dy
from dyrect import draw_complex, unit_circle_sample, EpsilonNet, NerveComplex, AlphaNerveComplex

np.random.seed(2)
# crc1 = unit_circle_sample(4000, 0.75) + [1.1,0]
# crc2 = unit_circle_sample(4000, 0.75) - [1.1,0]
# points = np.append(crc1, crc2, axis=0)
# eps=.25

points = np.random.random((10000,2))
eps=0.1

EN = EpsilonNet(eps, 0)
EN.fit(points)
lms = EN.landmarks
# points = np.random.random((10000,2))

# crc1 = unit_circle_sample(1000, 0.75) + [1.1,0]
# crc2 = unit_circle_sample(1000, 0.75) - [1.1,0]
# points = np.append(crc1, crc2, axis=0)

print("EN done")
# points = np.random.random((1200,2))

# plt.figure()
# plt.scatter(points[:,0], points[:,1])
# plt.show()

# fig = plt.figure(figsize=(15,10))
# fig = plt.figure()
fig = plt.figure(figsize=(22, 12))
rows = 1
cols = 2

ax = plt.subplot(rows, cols, 1)
plt.scatter(points[:,0], points[:,1], s=0.2)
# plt.scatter(series[:,0], series[:,1], s=0.2)
plt.scatter(lms[:,0], lms[:,1], s=5.2)

for lm in lms:
    crc = plt.Circle(lm, eps, color='r', alpha=0.05)
    ax.add_patch(crc)

# ax = plt.subplot(rows, cols, 2)
# nc = NerveComplex(lms, eps, 2, points=points)
# draw_complex(nc, fig=fig, ax=ax)
# # plt.show()

ax = plt.subplot(rows, cols, 2)
anc = AlphaNerveComplex(lms, eps, 2, points=points, patching=False, record_witnesses=True)
draw_complex(anc, fig=fig, ax=ax, vlabels=True)
non_witnesses = np.array([points[i, :] for i in anc.non_witnesses[2]])
print(anc.not_witnessed[2])
ax.scatter(non_witnesses[:, 0], non_witnesses[:, 1], color='k', s = 0.5)
plt.show()

from scipy.spatial.distance import cdist
from itertools import combinations
X,Y = np.mgrid[0:1:0.001, 0:1:0.001]
xy = np.vstack((X.flatten(), Y.flatten())).T
grid_dists = cdist(xy, lms, 'euclidean')
points_dists = cdist(points, lms, 'euclidean')

# draw_complex(anc, fig=fig, ax=ax, vlabels=True)
edge_wareas = dict()
edge_witnesses = dict()
verts = [4, 33, 44, 57]
# verts = [22, 23, 60, 67]
# limits = [[0.7, 1.], [0.6, 0.9]]
# verts = [3, 8, 37, 47]
# limits = [[0, 0.28], [0.4, 0.65]]
# verts = [5, 25, 51, 54]
# limits = [[0.18, 0.4], [0.07, 0.3]]

vcoords = np.array([lms[i] for i in verts])
def get_limits(vertices, offset):
    xmin, xmax, ymin, ymax = np.min(vcoords[:, 0]), np.max(vcoords[:,0]), np.min(vcoords[:,1]), np.max(vcoords[:, 1])
    xspan, yspan = xmax-xmin, ymax-ymin
    return [[xmin - xspan*offset, xmax + xspan*offset], [ymin - yspan*offset, ymax + yspan*offset]]
limits = get_limits(vcoords, 0.2)

for edge in combinations(verts, 2):
    edge_wareas[tuple(np.sort(edge))] = []
    edge_witnesses[tuple(np.sort(edge))] = []
for x_arg_sorted, x in zip(np.argsort(grid_dists, axis=1), range(len(xy))):
    wit_edge = tuple(np.sort(x_arg_sorted[:2]))
    if wit_edge in edge_wareas:
        edge_wareas[wit_edge].append(x)
for x_arg_sorted, x in zip(np.argsort(points_dists, axis=1), range(len(points))):
    wit_edge = tuple(np.sort(x_arg_sorted[:2]))
    if wit_edge in edge_witnesses:
        edge_witnesses[wit_edge].append(x)
# print(edge_witnesses)

fig = plt.figure()
ax = plt.subplot()
for points_list in edge_wareas.values():
    if len(points_list) > 0:
        e_xy = np.array([xy[i, :] for i in points_list])
        ax.scatter(e_xy[:, 0], e_xy[:, 1])
ax.scatter(lms[:, 0], lms[:, 1], color='k', s = 10.5)
ax.scatter(points[:, 0], points[:, 1], color='k', s = 0.5)
plt.xlim(limits[0])
plt.ylim(limits[1])
plt.show()

fig = plt.figure()
ax = plt.subplot()
for points_list in edge_witnesses.values():
    if len(points_list) > 0:
        e_xy = np.array([points[i, :] for i in points_list])
        ax.scatter(e_xy[:, 0], e_xy[:, 1])
ax.scatter(lms[:, 0], lms[:, 1], color='k', s = 10.5)
ax.scatter(points[:, 0], points[:, 1], color='k', s = 0.5)
plt.xlim(limits[0])
plt.ylim(limits[1])
plt.show()

fig = plt.figure()
ax = plt.subplot()
anc2 = AlphaNerveComplex(lms, eps, 2, points=points)
draw_complex(anc2, fig=fig, ax=ax)
plt.show()


# ax = plt.subplot(rows, cols, 3)
# TM = dy.TransitionMatrix(lms, eps, alpha=True)
# transitions = TM.fit(points)
# prob_matrix = dy.trans2prob(transitions)
# dy.draw_transition_graph(prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)
#
# ax = plt.subplot(rows, cols, 4)
# GTM = GeomTransitionMatrix(lms, nc, eps, alpha=True)
# transitions = GTM.fit(series_emb)
# gtm_prob_matrix = dy.trans2prob(transitions)
# dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)