from .poset import Poset
from .epsilon_net import Complex
from disjoint_set import DisjointSet
from itertools import combinations
import numpy as np

class MVF:

    def __init__(self):
        self._fspace = Poset()
        self._partition = []
        self._order_complex = None
        self._point2simplex = dict()
        self._simplex2mv = dict()

    @property
    def fspace(self):
        return self._fspace

    @property
    def partition(self):
        return self._partition

    @property
    def complex(self):
        return self._order_complex

    def simplex2mv(self, s):
        return self._simplex2mv[s]

    @classmethod
    def from_cell_complex(cls, graph, cell_complex):
        """
        :param graph: transition graph between cells of a complex
        :param cell_complex: in the form of a nerve complex
        :return:
        """
        obj = cls.__new__(cls)

        ncells = len(cell_complex.simplices[0])
        p_nerve, simplex2idx = cell_complex.to_poset(idx=True)

        # proto-mvf on nerve
        vecs = DisjointSet({x: x for x in range(p_nerve.npoints)})
        # (source, target) in the transition graph
        for (s, t) in zip(*np.nonzero(graph)):
            # print((s,t))
            assert s != t
            if s < t:
                vecs.union(t, simplex2idx[(s, t)])
            else:
                vecs.union(t, simplex2idx[(t, s)])

        def get_vec_of(x):
            for v in vecs.itersets(True):
                if vecs.find(x) == v[0]:
                    return v[1]

        # print(vecs)
        for level in [d for d in cell_complex.simplices.keys() if d > 1]:
            for simplex in cell_complex.simplices[level]:
                # simplices directly below the simplex, i.e. subsets of the simplex without a sinle element
                belows = combinations(simplex, level)
                # multivector representants to which simplex can be attached without breaking the convexity
                candidates = set()
                sid = simplex2idx[simplex]
                for b in belows:
                    bid = simplex2idx[b]
                    # print([sid], get_vec_of(bid))
                    # set to be checked
                    is_this_set_convex = get_vec_of(bid).union([sid])
                    if p_nerve.is_convex(is_this_set_convex):
                        candidates.add(vecs.find(bid))
                if len(candidates) == 1:
                    vecs.union(candidates.pop(), sid)
        print(vecs)
        vecs_ids = [vecs.find(v.pop()) for v in vecs.itersets()]
        print(vecs_ids)
        print(simplex2idx)

        # build order complex/poset
        if cell_complex.coordinates is None:
            order_simplices = p_nerve.order_complex()
            order_complex = Complex.construct(order_simplices)
        else:
            # assumption to not mess up with indices - baricentric_subdivision() method uses order_complex() as well
            order_complex = cell_complex.baricentric_subdivision()

        # reverse poset
        # poset of the order complex and index map
        order_poset, idx2oidx = order_complex.to_poset(idx=True)


        # a map translating original cells (vertices in the input complex) to indices in the final poset
        cell2idx = dict()
        for i in range(ncells):
            idx = simplex2idx[tuple([i,])]
            oidx = idx2oidx[tuple([idx,])]
            # print(i, oidx)
            cell2idx[i] = oidx
        #

        mvf = {v:set() for v in vecs_ids}
        # order complex simplices
        oc_points = set(range(order_poset.npoints))
        for d in cell_complex.simplices.keys():
            for sigma in cell_complex.simplices[d]:
                idx_poset = simplex2idx[sigma]
                idx_oposet = idx2oidx[tuple([idx_poset,])]
                upper_set = order_poset.above(idx_oposet)
                v_points = oc_points.intersection(upper_set)
                oc_points = oc_points.difference(v_points)
                # print(idx_poset, " - idx - ", vecs.find(idx_poset))
                # print(v_points.union(mvf[vecs.find(idx_poset)]), vecs.find(idx_poset))
                mvf[vecs.find(idx_poset)] = mvf[vecs.find(idx_poset)].union(v_points)

        # for mv in mvf:
        #     print([idx2])

        # for v in vecs:
        #     mv = set()
        #     for sigma in v:
        #         mv = mv.union(order_poset.above(sigma).intersection)


        obj._point2simplex = {v: k for k,v in idx2oidx.items()}
        obj._fspace = order_poset
        obj._partition = list(mvf.values())
        obj._order_complex = order_complex

        osimplex2mv = dict()
        for mv_idx, mv in enumerate(obj._partition):
            for p in mv:
                osimplex2mv[obj._point2simplex[p]] = mv_idx
        obj._simplex2mv = osimplex2mv


        # return order_complex, order_poset, list(mvf.values()), oidx2osimplex
        return obj
        # using opening compute the final multivector field

    # @classmethod
    # def from_complex(cls, partition):