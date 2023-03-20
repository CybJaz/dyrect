import unittest

import math
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import dyrect
from dyrect.data_generators import *


class MyTestCase(unittest.TestCase):
    def test_four_winged_generator(self):
        n=20000
        points=dadras_attractor(n, starting_point=[1., 1., 1., 1.], skip=10000, adaptive_step=False, step=0.01)

        self.assertEqual(len(points), n)
        self.assertFalse(any(points[-1]==0.))
        self.assertFalse(any([math.isnan(x) for x in points[-1]]))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
        plt.show()

    def test_dadras_generator(self):
        n=20000
        points=dadras_attractor(n, starting_point=[1., 10., 10., 1.], skip=10000, adaptive_step=False, step=0.01)

        self.assertEqual(len(points), n)
        self.assertFalse(any(points[-1]==0.))
        self.assertFalse(any([math.isnan(x) for x in points[-1]]))

        # fig = plt.figure()
        # # ax = fig.add_subplot()
        # # ax.plot(points[:, 0], points[:, 2], linewidth=.1)
        # # ax.scatter(points[:, 0], points[:, 2], s=0.1)
        # ax = fig.add_subplot(projection='3d')
        # ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=.1)
        # # ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.1)
        # plt.show()
        #
        # embx = dyrect.embedding(points[:, 0], 3, 8)
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.plot(embx[:, 0], embx[:, 1], linewidth=.1)
        # # ax = fig.add_subplot(projection='3d')
        # # ax.plot(embx[:, 0], embx[:, 1], embx[:, 2], linewidth=.1)
        # plt.show()

    def test_lorenz_generator(self):
        n=10000
        points=lorenz_attractor(n)

        self.assertEqual(len(points), n)
        self.assertFalse(any(points[-1]==0.))
        self.assertFalse(any([math.isnan(x) for x in points[-1]]))

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
        # plt.show()

    def test_lemniscate_generator(self):
        n = 10000
        points = lemniscate(n)

        self.assertEqual(points.shape[0], n)
        self.assertEqual(points.shape[1], 2)
        self.assertFalse(any(points[-1] == 0.))
        self.assertFalse(any([math.isnan(x) for x in points[-1]]))

# %    def test_limit_cycle_system(self):
#         nsp = 2000
#         ts = 5
#         step = 0.1
#
#         limx = 1.7
#         bounds = np.array([[-limx, limx], [-limx, limx]])
#
#         trajectories, starting_points = sampled_2d_system(double_limit_cycle, nsp, ts, step=step, bounds=bounds)
#         # trajectories = sampled_2d_system(limit_cycle, nsp, ts, step=step)
#         from collections import Counter
#         print(Counter([len(t) for t in trajectories]))
#
#         fig = plt.figure()
#         for traj in trajectories:
#             plt.plot(traj[:,0], traj[:,1])
#         plt.scatter(starting_points[:,0], starting_points[:, 1], s=1.5)
#         plt.show()
#

if __name__ == '__main__':
    unittest.main()
