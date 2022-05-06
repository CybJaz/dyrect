import matplotlib.pyplot as plt
# import miniball
import numpy as np
# import scipy as sc

from dyrect import draw_complex, unit_circle_sample, EpsilonNet, NerveComplex, AlphaNerveComplex

np.random.seed(2)
# crc1 = unit_circle_sample(20000, 0.5) + [1.1,0]
# crc2 = unit_circle_sample(20000, 0.5) - [1.1,0]
# points = np.append(crc1, crc2, axis=0)

points = np.random.random((1000,2))

eps=.2
EN = EpsilonNet(eps, 0)
EN.fit(points)
lms = EN.landmarks

# points = np.random.random((1200,2))

# plt.figure()
# plt.scatter(points[:,0], points[:,1])
# plt.show()

fig = plt.figure(figsize=(15,10))
rows = 2
cols = 2

ax = plt.subplot(rows, cols, 1)
plt.scatter(points[:,0], points[:,1], s=0.2)
# plt.scatter(series[:,0], series[:,1], s=0.2)
plt.scatter(lms[:,0], lms[:,1], s=5.2)

for lm in lms:
    crc = plt.Circle(lm, eps, color='r', alpha=0.05)
    ax.add_patch(crc)

ax = plt.subplot(rows, cols, 2)
nc = NerveComplex(lms, eps, 2, points=points)
draw_complex(nc, fig=fig, ax=ax)
# plt.show()

ax = plt.subplot(rows, cols, 3)
anc = AlphaNerveComplex(lms, eps, 2, points=points)
draw_complex(anc, fig=fig, ax=ax)
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