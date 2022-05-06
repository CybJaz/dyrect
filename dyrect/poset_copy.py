from scipy.sparse import csr_matrix
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
        # only direct succesors (lower elements)
        self._down = csr_matrix((n,n), dtype=np.int8)
        # only direct predecessors (highier elements)
        self._up = csr_matrix((n,n), dtype=np.int8)

        # all elements lower then the given one
        # self._closures = csr_matrix(np.diag(np.ones(n)), dtype=np.int8)
        self._closures = csr_matrix((n,n), dtype=np.int8)
        self._closures.setdiag(np.ones(n))
        # all elements highier then the given one
        self._openings = csr_matrix(np.diag(np.ones(n)), dtype=np.int8)


    # TODO: method for building poset from a complex
    # @classmethod
    # def from_complex(cls):

    # TODO: method for building a linear order
    # @classmethod
    # def linear_order(cls, n=1):

    @classmethod
    def from_down_graph(cls, down_graph):
        """
        Construct a poset from a down graph (relation of being smaller)
        :param down_graph: array-like structure
        :return:
        """
        obj = cls.__new__(cls)
        (nx, ny) = down_graph.shape
        assert nx == ny
        obj._npoints = nx
        obj._down = csr_matrix(down_graph)
        obj._up = csr_matrix(np.transpose(down_graph))
        obj._closures = shortest_path(obj.down_graph, directed=True, unweighted=True)
        obj._closures = csr_matrix(np.where(obj._closures == np.inf, 0, np.where(obj._closures > 0, 1,0)) + np.diag(np.ones(nx)))

        obj._openings = csr_matrix(np.transpose(obj._closures))
        #     shortest_path(obj.up_graph, directed=True, unweighted=True)
        # obj._openings = np.where(obj._openings == np.inf, 0, np.where(obj._openings > 0, 1,0)) + np.diag(np.ones(nx))
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

    @property
    def closures(self):
        return self._closures

    @property
    def openings(self):
        return self._openings

    def add_relation(self, lx, hx):
        """
        Add a relation lx < hx
        :param lx:
        :param hx:
        """
        assert self.closures[lx, hx] != 1, 'this relation will break the partial order'

        if self.closures[hx, lx] == 0:
            self._down[hx, lx] = 1
            self._up[lx, hx] = 1
            print(self.closures.tolil().rows[lx])
            # if something is below lx then it will be below hx as well
            for x in self.closures.tolil().rows[lx]:
                self.closures[hx, x] = 1
            # if something is below lx then it will be below hx as well
            for x in self.openings.tolil().rows[lx]:
                self.openings[lx, x] = 1
        # otherwise we already have lx < hx