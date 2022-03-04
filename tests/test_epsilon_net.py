import numpy as np
import dyrect.epsilon_net as en
import dyrect.reconstruction as tg
from dyrect.data_generators import lemniscate
from dyrect.drawing import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

np.random.seed(0)
points = lemniscate(np.linspace(0, 200 * np.pi, num=2000)) + \
         np.random.random_sample((2000, 2)) * 0.2

eps = 0.2
EN = en.EpsilonNet(eps, 0)
EN.fit(points)
lms = EN._landmarks
print(lms.shape)

plt.figure()
plt.scatter(points[:, 0], points[:, 1], s=0.5)
plt.scatter(lms[:, 0], lms[:, 1], s=21.9)

TM = tg.TransitionMatrix(lms, eps)
transitions = TM.fit(points)
prob_matrix = tg.trans2prob(transitions)

draw_transition_graph(prob_matrix, lms, threshold=0.15)
plt.show()

# DG = nx.DiGraph()
#
# threshold = 0.2
# nnodes = len(transitions)
#
# for i in range(nnodes):
#     edges = [(i, j, prob_matrix[i, j]) for j in range(nnodes) if prob_matrix[i, j] > threshold]
#     #     print(edges)
#     DG.add_weighted_edges_from(edges)
# # for i in range(nnodes):
# #     if prob_matrix[i,i] > threshold:
# #         print([i,prob_matrix[i,i]])
#
# edge_colors = [e[2] for e in DG.edges.data("weight")]
# cmap = plt.cm.plasma
#
# plt.figure(figsize=(10, 6))
# nx.draw_networkx_nodes(DG, pos=lms, node_size=50)
# edges = nx.draw_networkx_edges(DG, pos=lms, node_size=50, edge_color=edge_colors, edge_cmap=cmap, width=2,
#                                arrowsize=10);
#
# pc = mpl.collections.PatchCollection(edges, cmap=cmap)
# pc.set_array(edge_colors)
# plt.colorbar(pc)
# ax = plt.gca()
# ax.set_axis_off()
# plt.show()

# def unit_circle_sample(npoints, noise=0.0):
#     rpoints = 2 * np.pi * np.random.random_sample((npoints))
#     x = np.cos(rpoints) + (np.random.random_sample((npoints)) - 0.5) * noise
#     y = np.sin(rpoints) + (np.random.random_sample((npoints)) - 0.5) * noise
#     return np.transpose(np.stack((x, y)))
#
# crc1 = unit_circle_sample(1000, 0.5) + [1.1,0]
# crc2 = unit_circle_sample(5000, 0.5) - [1.1,0]
# points = np.append(crc1, crc2, axis=0)
# print(points.shape)
# EN = en.EpsilonNet(0.2, 0)
# EN.fit(points)
# lms = EN.landmarks_
# print(lms.shape)
#
# plt.figure()
# plt.scatter(points[:,0],points[:,1], s=0.5)
# plt.scatter(lms[:,0],lms[:,1], s=21.9)
# plt.show()
