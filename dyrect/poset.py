from itertools import product
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np


class Poset:
    def __init__(self, n=1):
        """
        :param n: number of points in the poset
        """
        assert n >= 1
        # number of points in the poset
        self._npoints = n

        self._down = lil_matrix((n,n), dtype=np.int8)
        self._down.setdiag(np.ones(n))
        self._up = lil_matrix((n, n), dtype=np.int8)
        self._up.setdiag(np.ones(n))

    # TODO: method for building poset from a complex
    # @classmethod
    # def from_complex(cls):

    # TODO: method for building a linear order
    # @classmethod
    # def linear_order(cls, n=1):

    @classmethod
    def from_down_graph(cls, down_graph):
        assert isinstance(down_graph, lil_matrix)
        obj = cls.__new__(cls)
        obj._npoints = down_graph.shape[0]
        obj._down = down_graph
        obj._up = lil_matrix(np.transpose(obj._down), dtype=np.int8)
        return obj

    @classmethod
    def from_dag(cls, dag):
        """
        Construct a poset from a graph (relation of being smaller)
        :param dag: directed acyclic graph in the form of an array-like structure of shape (n,n)
        :return:
        """
        obj = cls.__new__(cls)
        (nx, ny) = dag.shape
        assert nx == ny
        obj._npoints = nx
        obj._down = lil_matrix(shortest_path(dag, directed=True, unweighted=True), dtype=np.int8)
        for (ix,iy) in zip(*obj._down.nonzero()):
            if obj._down[ix,iy] == np.inf:
                obj._down[ix, iy] = 0
            elif obj._down[ix,iy] > 0:
                obj._down[ix, iy] = 1
        for i in range(nx):
            obj._down[i, i] = 1

        obj._up = lil_matrix(np.transpose(obj._down), dtype=np.int8)
        return obj

    @property
    def npoints(self):
        return self._npoints

    @property
    def down_graph(self):
        return self._down

    @property
    def up_graph(self):
        return self._up

    def below(self, x):
        return set(self._down.getrow(x).nonzero()[1])

    def above(self, x):
        return set(self._up.getrow(x).nonzero()[1])

    def succesors(self, x):
        succ = set(self._down.getrow(x).nonzero()[1]).difference([x])
        not_used = list(succ.copy())
        # print("X: ", x, not_used, succ)

        while len(not_used) > 0:
            y = not_used.pop(-1)
            below_y = set(self._down.getrow(y).nonzero()[1])
            not_used = list(set(not_used).difference(below_y))
            succ = succ.difference(below_y.difference(set([y])))
            # print(y, below_y, not_used, succ)

        return succ

    def add_relation(self, lx, hx):
        """
        Add a relation lx < hx
        :param lx:
        :param hx:
        """
        assert self.down_graph[lx, hx] == 0, 'this relation will break the partial order'

        if self.down_graph[hx, lx] == 0:
        # otherwise we already have lx < hx
            self.down_graph[hx, lx] = 1
            self.up_graph[lx, hx] = 1
            belows = self._down.getrow(lx).nonzero()[1]
            aboves = self._up.getrow(hx).nonzero()[1]

            # if something is below lx then it will be below everything above hx as well
            for (b,a) in product(belows, aboves):
                self._down[a, b] = 1
                self._up[b, a] = 1

    def get_reversed(self):
        return Poset.from_down_graph(self._up)

    def is_convex(self, s):
        """
        Checks if the given set is convex in the poset
        :param s: a set
        :return: True or False
        """
        assert min(s) >= 0 and max(s) < self.npoints
        # original set S
        ss = set(s)
        # the part of a set S that hasn't been checked yet
        sa = ss.copy()
        # the part of the mouth already tested
        sm = set()
        while len(sa) > 0:
            x = sa.pop()
            down_set = set(self._down[x].nonzero()[1])
            # new part of the mouth
            dm = down_set.difference(ss.union(sm))
            # print(x, sa, sm, dm)
            sa = sa.difference(down_set)
            for y in dm:
                # print(y, set(self._down[y].nonzero()[1]))
                if len(set(self._down[y].nonzero()[1]).intersection(ss)) > 0:
                    return False
            sm = sm.union(dm)
        return True

    def order_complex(self):
        """
        Generate an order complex of a poset.
        :return:
        """
        simplices = dict()
        simplices[0] = []

        # we are doing a depth search to generate all possible paths
        stack = [[x] for x in range(self.npoints)]
        while len(stack)>0:
            path = stack.pop()
            d = len(path)-1
            # TODO: we could track the max depth of the poset during the construction
            if d not in simplices:
                simplices[d] = []
            # print(path)
            simplices[d].append(tuple(np.sort(path)))
            x = path[-1]
            for y in self._down[x].nonzero()[1]:
                if x != y:
                    # print("Appending: ", path + [y])
                    stack.append(path + [y])

        return simplices
