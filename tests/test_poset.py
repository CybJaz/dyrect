from dyrect import Poset, draw_poset
import numpy as np
import unittest
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

def eq(a1, a2):
    if len(a1)!=len(a2):
        print("diff size")
        return False
    else:
        return all([a1[i]==a2[i] for i in range(len(a1))])



class TestPoset(unittest.TestCase):
    def test_poset_init(self):
        empty_poset = Poset()
        self.assertTrue(empty_poset.up_graph.shape == (1,1))
        self.assertTrue(empty_poset.down_graph.shape == (1, 1))
        self.assertEqual(empty_poset.up_graph[0,0], 1)
        self.assertEqual(empty_poset.down_graph[0,0], 1)

        nonempty_poset = Poset(10)
        self.assertTrue(nonempty_poset.up_graph.shape == (10,10))
        self.assertTrue(nonempty_poset.down_graph.shape == (10, 10))

        # self.assertEqual(nonempty_poset.closures[1,1], 1)
        # self.assertEqual(nonempty_poset.closures[1,0], 0)
        # self.assertEqual(nonempty_poset.openings[1,1], 1)
        # self.assertEqual(nonempty_poset.openings[0,1], 0)

        # self.assertTrue(isinstance(nonempty_poset.closures, csr_matrix))
        # self.assertTrue(isinstance(nonempty_poset.openings, csr_matrix))
        self.assertTrue(isinstance(nonempty_poset.up_graph, lil_matrix))
        self.assertTrue(isinstance(nonempty_poset.down_graph, lil_matrix))

    def test_poset_from_down_graph(self):
        # print("CREATE A POSET FROM A DOWN-GRAPH")
        dg = np.array([[1,0,0,0], [1,1,0,0], [1,0,0,0], [0,1,1,0]])
        poset = Poset.from_dag(dg)
        # print(poset.down_graph.toarray())
        # print(poset.closures)
        self.assertTrue(isinstance(poset.down_graph, lil_matrix))
        self.assertTrue(isinstance(poset.up_graph, lil_matrix))
        # print(poset.down_graph.toarray()[0])
        self.assertTrue(eq(poset.down_graph.toarray()[0], [1,0,0,0]))
        self.assertTrue(eq(poset.down_graph.toarray()[1], [1, 1, 0, 0]))
        self.assertTrue(eq(poset.down_graph.toarray()[2], [1, 0, 1, 0]))
        self.assertTrue(eq(poset.down_graph.toarray()[3], [1, 1, 1, 1]))


    def test_adding_relation(self):
        # print("ADDING RELATION")
        p = Poset(8)
        # print(p.down_graph.toarray())
        p.add_relation(6,7)
        p.add_relation(0,3)
        p.add_relation(1,3)
        p.add_relation(1,6)
        p.add_relation(0,4)
        p.add_relation(2,4)
        # print("step 1")
        # print("down graph")
        # print(p.down_graph.toarray())
        # print("up graph")
        # print(p.up_graph.toarray())
        p.add_relation(3,6)
        # print("step 2")
        # print("down graph")
        # print(p.down_graph.toarray())
        # print("up graph")
        # print(p.up_graph.toarray())
        p.add_relation(1,5)
        p.add_relation(2,5)
        p.add_relation(4,6)
        p.add_relation(5,6)
        # print("step 3")
        # print("down graph")
        # print(p.down_graph.toarray())
        # print("up graph")
        # print(p.up_graph.toarray())

        with self.assertRaises(AssertionError):
            assert 1 == 0
        with self.assertRaises(AssertionError):
            p.add_relation(6,2)

        self.assertTrue(eq(p.down_graph.getrow(6).toarray()[0], (1,1,1,1,1,1,1,0)))
        self.assertTrue(eq(p.up_graph.getrow(6).toarray()[0], (0,0,0,0,0,0,1,1)))

        self.assertTrue(eq(p.down_graph.getrow(2).toarray()[0], (0,0,1,0,0,0,0,0)))
        self.assertTrue(eq(p.up_graph.getrow(2).toarray()[0], (0,0,1,0,1,1,1,1)))

        self.assertTrue(eq(p.down_graph.getrow(3).toarray()[0], (1,1,0,1,0,0,0,0)))
        self.assertTrue(eq(p.up_graph.getrow(3).toarray()[0], (0,0,0,1,0,0,1,1)))
        self.assertTrue(eq(p.down_graph.getrow(5).toarray()[0], (0,1,1,0,0,1,0,0)))
        self.assertTrue(eq(p.up_graph.getrow(5).toarray()[0], (0,0,0,0,0,1,1,1)))

        p.order_complex()

    def test_convexity_test(self):
        p = Poset(8)
        # print(p.down_graph.toarray())
        p.add_relation(6,7)
        p.add_relation(0,3)
        p.add_relation(1,3)
        p.add_relation(1,6)
        p.add_relation(2,4)
        p.add_relation(0,4)
        p.add_relation(3,6)
        p.add_relation(1,5)
        p.add_relation(2,5)
        p.add_relation(4,6)
        p.add_relation(5,6)
        # print(p.down_graph.toarray())

        self.assertEqual(p.succesors(2), set())
        self.assertEqual(p.succesors(3), set([0,1]))
        self.assertEqual(p.succesors(6), set([3,4,5]))
        self.assertEqual(p.succesors(7), set([6]))

        self.assertTrue(not p.is_convex([0,3,6]))
        self.assertTrue(not p.is_convex([0, 6]))
        self.assertTrue(not p.is_convex([4, 2, 7]))
        self.assertTrue(not p.is_convex([5,2,7]))
        self.assertTrue(p.is_convex([0,3,4,6]))
        self.assertTrue(p.is_convex([4, 6, 7]))
        self.assertTrue(p.is_convex([1, 3]))
        self.assertTrue(p.is_convex([0, 1, 3]))
        self.assertTrue(p.is_convex([5, 3]))

        draw_poset(p)
        rp = p.get_reversed()
        draw_poset(rp)
        plt.show()

if __name__ == '__main__':
    unittest.main()
    # poset = Poset()