import unittest
import matplotlib.pyplot as plt
import numpy as np
from dyrect import Complex, draw_complex

class MyTestCase(unittest.TestCase):
    def test_simple_construction(self):
        c = Complex(simplices={0: [(0,), (1,), (2,), (3,), (4,), (5,)],
                               1: [(0, 1), (1, 2), (1, 3), (1, 4), (0, 2), (2, 4), (3, 5), (4, 5)],
                               2: [(0, 1, 2), (1, 2, 4)]},
                    coords = np.array([[0.0, 0.0],
                                       [1.0, 1.0],
                                       [-1.0, 1.0],
                                       [1.0, 2.0],
                                       [-1., 2.0],
                                       [0.0, 3.0]])
                    )
        self.assertEqual(c.dimension, 2)  # add assertion here
        self.assertEqual(c.ambient_dimension, 2)  # add assertion here
        self.assertEqual([1, 1, 0], c.betti_numbers)
        self.assertEqual(1, len(c.components.subsets()))

        sc = c.subcomplex([0, 1, 5, 2])
        self.assertEqual(len(sc.simplices[0]), 4)
        self.assertEqual(len(sc.simplices[1]), 3)
        self.assertEqual(len(sc.simplices[2]), 1)
        self.assertEqual(sc.dimension, 2)
        self.assertEqual(sc.ambient_dimension, 2)
        self.assertEqual([2, 0, 0], sc.betti_numbers)
        self.assertEqual(2, len(sc.components.subsets()))

        sc = sc.subcomplex([0, 1, 4, 5])
        self.assertEqual(3, len(sc.simplices[0]))
        self.assertEqual(1, len(sc.simplices[1]))
        self.assertEqual(0, len(sc.simplices[2]))
        self.assertEqual(sc.dimension, 1)
        self.assertEqual(sc.ambient_dimension, 2)
        self.assertEqual([2, 0], sc.betti_numbers)
        self.assertEqual(2, len(sc.components.subsets()))

        sc = c.subcomplex([0, 1, 4, 5])
        self.assertEqual(4, len(sc.simplices[0]))
        self.assertEqual(3, len(sc.simplices[1]))
        self.assertEqual(0, len(sc.simplices[2]))
        self.assertEqual(sc.dimension, 1)
        self.assertEqual(sc.ambient_dimension, 2)
        self.assertEqual([1, 0], sc.betti_numbers)
        self.assertEqual(1, len(sc.components.subsets()))

        # draw_complex(sc, vlabels=True)
        # plt.show()

if __name__ == '__main__':
    unittest.main()
    MyTestCase.simple_construction()
