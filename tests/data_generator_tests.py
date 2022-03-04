import unittest

import math
# import matplotlib.pyplot as plt
from dyrect.data_generators import *


class MyTestCase(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
