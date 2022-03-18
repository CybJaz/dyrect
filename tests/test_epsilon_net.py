import numpy as np
# import dyrect.epsilon_net as en
# import dyrect.reconstruction as tg
# from dyrect.data_generators import lemniscate
# from dyrect.drawing import *
import dyrect as dy
import matplotlib.pyplot as plt

bplot = True

np.random.seed(0)
points = dy.lemniscate(10000, step=0.2, tnoise=0.02, noise=0.05)

eps = 0.15
EN = dy.EpsilonNet(eps, 0)
EN.fit(points)
lms = EN._landmarks
print(lms.shape)

if bplot:
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=0.5)
    plt.scatter(lms[:, 0], lms[:, 1], s=21.9)

TM = dy.TransitionMatrix(lms, eps)
transitions = TM.fit(points)
prob_matrix = dy.trans2prob(transitions)

nc = dy.NerveComplex(lms, eps, 2, points)
print(nc.betti_numbers)

if bplot:
    dy.draw_transition_graph(prob_matrix, lms, threshold=0.15)
    plt.show()
    dy.draw_complex(nc)
    plt.show()

    snc, _ = nc.subcomplex(list(range(10,40)))
    dy.draw_complex(snc)
    plt.show()



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
