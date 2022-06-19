from .poset import Poset
from .epsilon_net import Complex
from gudhi import SimplexTree
from disjoint_set import DisjointSet
from itertools import combinations
import numpy as np
from math import inf

class MVF:

    def __init__(self):
        self._fspace = Poset()
        self._partition = []
        self._conley_index = []
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

    def conley_index(self, mv_idx):
        return self._conley_index[mv_idx]

    def is_critical(self, mv_idx):
        return np.sum(self._conley_index[mv_idx])>0

    @classmethod
    def from_cell_complex(cls, graph, cell_complex, assigning_style='balanced'):
        """
        :param graph: transition graph between cells of a complex
        :param cell_complex: in the form of a nerve complex
        :param,assigning_style:
            'prudent' - if it's not clear how to assign a simplex, do not assign it and create a new multivector
            'balanced' - assign a simplex to a multivector only if there exists unique multivector of a dimension highier than the other candidates
            'frivolous' - assign a simplex to any multivector among the candidates with the highiest dimension
            TODO: 'greedy' - merge multivectors whenever there is no clear way on how to assign a simplex
        :return:
        """
        obj = cls.__new__(cls)

        ncells = len(cell_complex.simplices[0])
        p_nerve, simplex2idx = cell_complex.to_poset(idx=True)

        # proto-mvf on nerve
        vecs = DisjointSet({x: x for x in range(p_nerve.npoints)})
        # (source, target) in the transition graph
        for (s, t) in zip(*np.nonzero(graph)):
            # assert s != t
            if s < t:
                vecs.union(simplex2idx[(t,)], simplex2idx[(s, t)])
                # print((s, t), simplex2idx[(s, t)])
            elif s > t:
                vecs.union(simplex2idx[(t,)], simplex2idx[(t, s)])
                # print((s, t), simplex2idx[(t,s)])

        def get_vec_of(x):
            # v = (key of a set, a set)
            for v in vecs.itersets(True):
                if vecs.find(x) == v[0]:
                    return v[1]

        # print("Proto MVF: ", vecs)
        # print("simplex2idx: ", simplex2idx)
        # idx2simplex is only TMP
        idx2simplex = {v: k for k, v in simplex2idx.items()}
        # for vs in [(v, [idx2simplex[x] for x in v])  for v in vecs.itersets()]:
        #     print(vs)

        for level in [d for d in cell_complex.simplices.keys() if d >= 1]:
            for simplex in cell_complex.simplices[level]:
                sid = simplex2idx[simplex]
                # if an edge is already assigned to a multivector then continue
                if level == 1 and len(get_vec_of(sid))>1:
                    continue
                # simplices directly below the simplex, i.e. subsets of the simplex without a single element
                belows = combinations(simplex, level)
                # multivector representants to which simplex can be attached without breaking the convexity
                candidates = set()
                for b in belows:
                    bid = simplex2idx[b]
                    # print([sid], get_vec_of(bid))
                    # set to be checked
                    is_this_set_convex = get_vec_of(bid).union([sid])
                    if p_nerve.is_convex(is_this_set_convex):
                        candidates.add(vecs.find(bid))
                if len(candidates) == 1:
                    vecs.union(candidates.pop(), sid)
                elif assigning_style != 'prudent' and len(candidates) > 1:
                    candidates = list(candidates)
                    # print(candidates)
                    # print([[idx2simplex[x]  for x in get_vec_of(v)] for v in candidates])
                    # the dimension of proto multivectors (0 is the highiest because the complex it is a nerve)
                    candidates_levels = [min([len(idx2simplex[x])-1 for x in get_vec_of(v)]) for v in candidates]
                    # print(candidates_levels)
                    min_candidates_levels = np.where(candidates_levels == np.min(candidates_levels))[0]
                    # print(min_candidates_levels)

                    if len(min_candidates_levels) == 1 and assigning_style == 'balanced':
                        vecs.union(candidates[min_candidates_levels[0]], sid)
                    elif assigning_style == 'frivolous':
                        vecs.union(candidates[min_candidates_levels[0]], sid)
        # print(vecs)
        vecs_ids = [vecs.find(v.pop()) for v in vecs.itersets()]
        # print(vecs_ids)

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

        # compute conley index of all multivectors
        obj._conley_index = []
        # obj._is_critical = []
        max_dim = 0
        for v in obj._partition:
            mth = obj._fspace.mouth(v)
            st = SimplexTree()
            # print(v, mth)
            # print([obj._point2simplex[m] for m in mth])
            if len(mth) > 0:
                for m in mth:
                    st.insert(obj._point2simplex[m], 0.)
            for x in v:
                st.insert(obj._point2simplex[x], 1.)
            homology = st.persistence()
            # print(homology)
            betti = dict()
            # TODO: check
            for h in homology:
                if h[1] == (1.0, inf):
                    if h[0] in betti:
                        betti[h[0]] += 1
                    else:
                        betti[h[0]] = 1
                    if h[0] > max_dim:
                        max_dim = h[0]
                elif h[1] == (0.0, 1.0):
                    if h[0]+1 in betti:
                        betti[h[0]+1] += 1
                    else:
                        betti[h[0]+1] = 1
                    if h[0]+1 > max_dim:
                        max_dim = h[0]+1

            obj._conley_index.append(tuple([betti[d] if d in betti else 0 for d in range(max_dim+1)]))
            # print(obj._conley_index[-1])
            # obj._is_critical.append()


         # print([mvf._point2simplex[s] for s in v])

        # return order_complex, order_poset, list(mvf.values()), oidx2osimplex
        return obj
        # using opening compute the final multivector field

    # @classmethod
    # def from_complex(cls, partition):
