import itertools

from gudhi import SimplexTree
from itertools import combinations
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.distance import cdist

from .poset import Poset
from itertools import chain, combinations
import functools

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

class EpsilonNet:
    def __init__(self, eps, max_num_of_landmarks=0, dist=True, method='weighted_furthest'):
        """
        :param eps:
        :param max_num_of_landmarks:
        :param dist: True - return distance matrix, False - return boolean matrix (is point covered by a cover element)
        :param method:
        """
        self._eps = eps
        self._max_num_of_landmarks = max_num_of_landmarks if max_num_of_landmarks > 0 else np.iinfo(np.int16).max
        self._return_distances = dist
        self._method = method
        self._nlandmarks = 0
        self._landmarks = []
        # self._complex = SimplexTree()

    def fit(self, X, Y=None):
        """ X - [number_of_samples, number_of_features] """
        nsamples, nfeatures = X.shape
        self._nlandmarks = 1
        self._landmarks = np.array([X[0]])

        distance_to_landmarks = np.array([np.array(np.linalg.norm(X - self._landmarks[0], axis=1))])
        distance_to_cover = distance_to_landmarks[0]
        while self._nlandmarks < self._max_num_of_landmarks and np.max(distance_to_cover) >= self._eps:
            if self._method == 'furthest_point':
                furthest_point_idx = np.argmax(distance_to_cover)
            elif self._method == 'weighted_furthest':
                distance_to_cover = [d if d >= self._eps else 0 for d in distance_to_cover]
                weights = np.power(distance_to_cover / np.max(distance_to_cover), 2)
                weights = weights / np.sum(weights)
                furthest_point_idx = np.random.choice(range(nsamples), p=weights)

            self._landmarks = np.append(self._landmarks, [X[furthest_point_idx]], axis=0)
            distance_to_landmarks = np.append(distance_to_landmarks,
                                              [np.array(np.linalg.norm(X - self._landmarks[self._nlandmarks], axis=1))],
                                              axis=0)
            distance_to_cover = np.min(np.stack((distance_to_cover, distance_to_landmarks[-1])), axis=0)
            self._nlandmarks += 1

        return np.transpose(distance_to_landmarks)

    @property
    def landmarks(self):
        return self._landmarks

    # @property
    # def complex(self):
    #     return self._complex

class Complex():
    def __init__(self, simplices = None, coords = None, max_dim=-1, ambient_dim=None):
        """
        @param simplices: dictionary of simplices; WARNING: the constructor does not check if faces are missing
        @param coords:
        @param max_dim:
        @param ambient_dim:
        """
        self._st = SimplexTree()
        if simplices is None:
            self._simplices = dict()
            self._betti_numbers = None
        else:
            self._simplices = simplices
            for d in simplices:
                for s in simplices[d]:
                    self._st.insert(s)
            self._st.compute_persistence(persistence_dim_max=True)
            self._betti_numbers = self._st.betti_numbers()

        self._coordinates = coords
        if coords is not None:
            self._ambient_dim = coords.shape[1]
        else:
            self._ambient_dim = ambient_dim

        dimension = [k for k in self._simplices.keys() if len(self._simplices[k]) == 0]
        if len(dimension) == 0:
            dimension = max(self._simplices.keys())
        else:
            dimension = min(dimension) - 1
        if max_dim < 0:
            self._dim = dimension
        else:
            self._dim = min(max_dim, dimension)

    @classmethod
    def construct(cls, simpls, coords=None, max_dim = -1):
        obj = cls.__new__(cls)
        obj.__init__()
        obj._simplices = simpls
        obj._coordinates = coords
        for ds in simpls:
            for s in simpls[ds]:
                obj._st.insert(s)
        obj._st.compute_persistence(persistence_dim_max=True)
        obj._betti_numbers = obj._st.betti_numbers()

        if coords is not None:
            obj._ambient_dim = coords.shape[0]
        dimension = [k for k in simpls.keys() if len(simpls[k]) == 0]
        if len(dimension) == 0:
            dimension = max(simpls.keys())
        else:
            dimension = min(dimension) - 1
        if max_dim < 0:
            obj._dim = dimension
        else:
            obj._dim = min(max_dim, dimension)
        return obj

    def add_simplex(self, simplex):
        d = len(simplex)-1
        if d not in self._simplices:
            self._simplices[d] = []
            self._dim = d
        self._simplices[d].append(simplex)
        self._st.insert(simplex)
        self._betti_numbers = None

    def merge_complex(self, patch):
        self._betti_numbers = None
        for d in patch.simplices:
            for s in patch.simplices[d]:
                if s not in self.simplices[d]:
                    self._simplices[d].append(s)
                    self._st.insert(s)

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def simplices(self):
        return self._simplices

    @property
    def ambient_dimension(self):
        return self._ambient_dim

    @property
    def dimension(self):
        return self._dim

    @property
    def nvertices(self):
        return len(self.coordinates)

    def to_poset(self, idx=False):
        """
        Get the face-poset representation of a complex
        :param idx: return the map between simplices and indices of points in the poset
        :return: Poset representing a complex
        """
        nsimplices = sum([len(x) for x in self.simplices.values()])
        poset = Poset(nsimplices)
        sim2idx = dict()
        for i, s in zip(range(nsimplices), chain(*self.simplices.values())):
            # print(i, s)
            sim2idx[s] = i
            if len(s) > 1:
                for face in combinations(s, len(s) - 1):
                    poset.add_relation(sim2idx[face], i)
        if idx:
            return poset, sim2idx
        else:
            return poset

    def baricentric_subdivision(self):
        poset, sim2idx = self.to_poset(idx=True)

        nsimplices = sum([len(x) for x in self.simplices.values()])
        baricenters = []
        # poset = Poset(nsimplices)
        # sim2idx = dict()
        # simplex2baricenter = dict()
        for i, s in zip(range(nsimplices), chain(*self.simplices.values())):
            # sim2idx[s] = i
            bc = np.average(self.coordinates[list(s), :], axis=0)
            baricenters.append(bc)
            # for d in range(1,len(s)):
            # if len(s) > 1:
            #     for face in combinations(s, len(s) - 1):
            #         poset.add_relation(sim2idx[face], i)
        baricentric_simplices = poset.order_complex()
        # print(baricentric_simplices)
        return Complex.construct(baricentric_simplices, np.array(baricenters))

    @property
    def components(self):
        """
            Get a partition of vertices into connected components.
        """
        components = DisjointSet([v for (v,) in self._simplices[0]])
        for (v1, v2) in self._simplices[1]:
            components.merge(v1, v2)
        return components

    def subcomplex(self, vertices):
        """
            Max subcomplex spanned by a set of vertices.
        """
        vset = set(vertices)

        sub_simplices = dict()
        sst = SimplexTree()
        components = DisjointSet(vset)

        for d in range(self.dimension + 1):
            sub_simplices[d] = []

        ### TODO: this could be done more efficiently
        for v in vset:
            if (v,) not in self._simplices[0]:
                continue
            cofaces = [c[0] for c in self._st.get_cofaces([v], 0)]
            for cof in cofaces:
                if set(cof).issubset(vertices):
                    d = len(cof) - 1
                    if tuple(cof) not in sub_simplices[d]:
                        sub_simplices[d].append(tuple(cof))
                        sst.insert(list(cof))
                    if d == 1:
                        components.merge(cof[0], cof[1])
        sst.persistence()

        return Complex(simplices=sub_simplices, coords=self.coordinates)

    @property
    def betti_numbers(self):
        if self._st is None:
            self._st = SimplexTree()
            max_d = -1
            for dsimplices in self._simplices.values():
                for s in dsimplices:
                    self._st.insert(list(s))
                    if len(s) > max_d:
                        max_d = len(s)
            self._st.set_dimension(max_d)
            self._st.compute_persistence(persistence_dim_max=True)

        if self._betti_numbers is None:
            self._st.compute_persistence(persistence_dim_max=True)
            self._betti_numbers = self._st.betti_numbers()
        return self._st.betti_numbers()

class NerveComplex(Complex):
    def __init__(self, lms, eps, max_dim, points=[]):
        """
        :param lms:
        :param eps:
        :param max_dim:
        :param points:
        """
        """
            TODO: if list of points is empty then generate just a rips-complex
            TODO: adjust code for max_dim=-1
        """
        self._st = SimplexTree()
        self._simplices = dict()
        self._coordinates = lms
        self._eps = eps
        self._dim = max_dim
        self._ambient_dim = lms.shape[1]

        assert False, "this class needs to be fixed"
        num_of_points = len(points)
        distances = cdist(points, lms, 'euclidean')

        assert num_of_points > 0, \
            "nerve_complex; the case with no points is not handled yet"

        for d in range(max_dim + 1):
            self._simplices[d] = []

        if len(points > 0):
            for i in range(len(points)):
                simplex = tuple(np.where(distances[i, :] <= eps * 1.15)[0])
                sdim = len(simplex) - 1
                if sdim == -1:
                    print(i, ': ', points[i], ' ', np.where(distances[i, :] <= eps * 1.))

                for d in range(min(max_dim + 1, sdim + 1)):
                    for s in combinations(simplex, d + 1):
                        if s not in self._simplices[d]:
                            self._simplices[d].append(s)
                        self._st.insert(list(s))
        else:
            print("TODO")
        self._st.persistence()

    @classmethod
    def construct(cls, st, simpls, coords, eps, dim):
        obj = cls.__new__(cls)
        obj._st = st
        obj._simplices = simpls
        obj._coordinates = coords
        obj._eps = eps
        obj._dim = dim
        return obj

    def subcomplex(self, vertices):
        """
            A clique subcomplex induced by a set of vertices.
        """
        vset = set(vertices)

        sub_simplices = dict()
        sst = SimplexTree()
        components = DisjointSet(vset)

        for d in range(self.dimension + 1):
            sub_simplices[d] = []

        for v in vset:
            cofaces = [c[0] for c in self._st.get_cofaces([v], 0)]
            for cof in cofaces:
                if set(cof).issubset(vertices):
                    d = len(cof) - 1
                    if tuple(cof) not in sub_simplices[d]:
                        sub_simplices[d].append(tuple(cof))
                        sst.insert(list(cof))
                    if d == 1:
                        components.merge(cof[0], cof[1])
        sst.persistence()

        # return NerveComplex.construct(sst, sub_simplices, np.array([self.coordinates[i] for i in vertices]), self._eps,
        #                               self.dimension), components.subsets()

        return NerveComplex.construct(sst, sub_simplices, self.coordinates, self._eps,
                                      self.dimension), components.subsets()

    @property
    def betti_numbers(self):
        if self._st is None:
            self._st = SimplexTree()
            max_d = -1
            for dsimplices in self._simplices.values():
                for s in dsimplices:
                    self._st.insert(list(s))
                    if len(s) > max_d:
                        max_d = len(s)
            self._st.set_dimension(max_d)
            self._st.compute_persistence()

        return self._st.betti_numbers()

    # @property
    # def coordinates(self):
    #     return self._coordinates
    #
    # @property
    # def simplices(self):
    #     return self._simplices
    #
    # @property
    # def dimension(self):
    #     return self._dim

    @property
    def epsilon(self):
        return self._eps


class OldPatchedWitnessComplex(NerveComplex):
    def __init__(self, lms, eps, max_dim=-1, points=[], patching=True,
                 patching_level=1, patched_dimensions=[2], record_witnesses=False):
        """
        :param lms: landmarks
        :param eps:
        :param max_dim:
        :param points:
        """
        # NerveComplex.__init__(self, lms, eps, max_dim, points=points)

        self._st = None
        self._simplices = dict()
        self._coordinates = lms
        self._eps = eps

        num_of_points = len(points)
        assert num_of_points > 0, \
            "alpha nerve_complex requires data points for construction"

        # print(len(points[0]))
        self._dim = len(points[0]) if max_dim==-1 else min(max_dim, len(points[0]))

        # distances from points to landmarks
        distances = cdist(points, lms, 'euclidean')

        # for each point get sorted (increasing, by the distance) list of landmarks no further than
        # epsilon times coefficient
        eps_lms = []
        for x_arg_sorted, x in zip(np.argsort(distances, axis=1), range(num_of_points)):
            i = 0
            while distances[x, x_arg_sorted[i]] < eps * 2.5:
                i += 1
            eps_lms.append(x_arg_sorted[:i])

        for d in range(self.dimension + 1):
            self._simplices[d] = []

        # 0-dimension; take centers of all cover elements that has non-empty its alpha cell part
        for x_eps_lms in eps_lms:
            assert len(x_eps_lms) >= 1
            new_simplex = tuple(x_eps_lms[:1],)
            if new_simplex not in self._simplices[0]:
                self._simplices[0].append(new_simplex)

        self._unproductive_witnesses = {0: [], 1: [], 2: [], 3: []}

        # 1 to d-dimension;
        # for dimension 2 - add a 2-simplex (a,b) if there exists a point x such that the list of
        #   the closest landmarks starts either with (a,b,...) or (b,a,...)
        # for dimension 3 - add a 3-simplex (a,b,c) if there exists a point x such that the list of
        #   the closest landmarks starts with (a,b,c), e.g. (b,a,c,d,f,e,...),
        #   and all faces of (a,b,c) are already in the complex
        for d in range(1, self.dimension + 1):
            # print("Dimension: ", d)
            for x, x_eps_lms in enumerate(eps_lms):
                if len(x_eps_lms) >= d+1:
                    new_simplex = tuple(np.sort(x_eps_lms[:d+1]))
                    # check if all boundary elements are already present
                    bdQ = True
                    for b in range(d+1):
                        boundary_simplex = new_simplex[:b] + new_simplex[b+1:]
                        if boundary_simplex not in self._simplices[d-1]:
                            self._unproductive_witnesses[d].append(x)
                            bdQ = False
                            break
                    if bdQ and (new_simplex not in self._simplices[d]):
                        self._simplices[d].append(new_simplex)

        if record_witnesses:
            self.non_witnesses = {0: [], 1: [], 2: [], 3:[]}
            self.not_witnessed = {0: [], 1: [], 2: [], 3:[]}
            # print(self._simplices[2])
            for x_arg_sorted, x in zip(np.argsort(distances, axis=1), range(num_of_points)):
                for d in range(self.dimension+1):
                    witnessed_simplex = tuple(np.sort(x_arg_sorted[:d+1]))
                    # print(witnessed_simplex)
                    if witnessed_simplex not in self._simplices[d]:
                        self.non_witnesses[d].append(x)
                        if witnessed_simplex not in self.not_witnessed[d]:
                            self.not_witnessed[d].append(witnessed_simplex)
            # print(self.non_witnesses[2])

        ##### SURFACE PATCHING #####
        if patching:
            # ### Patched dimension
            # pd = 2
            for pd in patched_dimensions:
                ### Co-dimension of the patching simplex
                for cdp in range(1, patching_level + 1):
                    # cdp = 1
                    ### Number of vertices of the patching simplex
                    nvp = pd + cdp + 1

                    extra_simplices = []
                    for x in self._unproductive_witnesses[pd]:
                        if len(eps_lms[x]) >= nvp:
                            new_simplex = tuple(np.sort(eps_lms[x][:nvp]))
                            if new_simplex in extra_simplices:
                                continue

                            ### Check if a new simplex fills a simple hole
                            bdQ = True
                            for boundary_simplex in combinations(new_simplex, pd + 1):
                                if boundary_simplex in self.simplices[pd]:
                                    bdQ = False
                                    break
                            if not bdQ:
                                continue
                            # print(new_simplex, bdQ)

                            ### count simplices of dimension pd-2 in the boundary of the patching simplex
                            count_bds = dict()
                            ### simplices of dimension pd-1
                            boundary = list()
                            ### new_simplex is of length nvp
                            for s in combinations(new_simplex, pd-1):
                                count_bds[s] = 0

                            for boundary_simplex in combinations(new_simplex, pd):
                                if boundary_simplex in self.simplices[pd-1]:
                                    boundary.append(boundary_simplex)
                                    for bbs in combinations(boundary_simplex, pd-1):
                                        count_bds[bbs] += 1

                            ### If it does, do the patching
                            if all([b == 2 or b == 0 for b in count_bds.values()]):
                                extra_simplices.append(tuple(new_simplex))

                                ### MORSE PATCHING
                                hole_boundary = set()
                                for s in boundary:
                                   hole_boundary = hole_boundary.union(set(powerset(s)))
                                hole_boundary = sorted(list(hole_boundary))
                                # print(hole_boundary)

                                patch = morse_patching(new_simplex, hole_boundary, self.coordinates)
                                # print(patch)
                                for p in patch:
                                    self._simplices[len(p)-1].append(p)

def patch_poset(simplex, boundary):
    patch = set(powerset(simplex))
    bd = set(chain.from_iterable([powerset(s) for s in boundary]))
    patch = patch.difference(bd)
    i2s = { i: s for i,s in enumerate(patch)}
    s2i = { s: i for i,s in enumerate(patch)}
    poset = Poset(len(i2s))
    for s in s2i:
        if len(s) == 1:
            continue
        for bs in combinations(s, len(s)-1):
            if bs in patch:
                poset.add_relation(s2i[bs], s2i[s])
    return poset, i2s, s2i


def morse_patching(simplex, boundary, verts):
    poset, i2s, s2i = patch_poset(simplex, boundary)

    i2diams = dict()
    for i in i2s.keys():
        i2diams[i] = []
        for e in combinations(i2s[i], 2):
            i2diams[i].append(np.linalg.norm(verts[e[0]] - verts[e[1]]))
        i2diams[i].sort(reverse=True)
    # print(i2diams)

    def diamcheck(x, y):
        """ given indices of two simplices x and y, check which:
            1) has higher dimension
            2) has longer edges
        """
        dx = i2diams[x]
        dy = i2diams[y]
        if len(dx) > len(dy):
            return 1
        elif len(dx) < len(dy):
            return -1
        else:
            for i in range(len(dx)):
                if dx[i] > dy[i]:
                    return 1
                elif dx[i] < dy[i]:
                    return -1
        return 0

    filling = set(range(len(i2s)))
    #     it = 0
    while True:
        old_filling = filling
        ### sorting of simplices for reduction with respect to:
        ## #1 method diamcheck
        sim_queue = sorted(list(filling), key=functools.cmp_to_key(diamcheck), reverse=True)
        ## #2 without sorting
        #         sim_queue = filling
        #     print(sim_queue)
        for i in sim_queue:
            up = (poset.above(i)).intersection(filling)
            #             it += 1
            if len(up) == 2:
                filling = filling.difference(up)
        #                 print([[i2s[s] for s in up]])
        #### This break makes it slightly faster for small cases but
        #### introduces more computations with bigger holes (check 'it' counter)
        #### but without the break the order might be altered
        #                 break

        if old_filling == filling:
            # print(True)
            break
    #     print(it)
    return [i2s[s] for s in filling]

# def morse_patching():


            #### END FOR
        # # find two the most distant vertices:
        # v_coords = np.array([self.coordinates[i] for i in new_simplex])

        #                 v_max = []
        #                 max_dist = -1
        #                 min_dist = 10000000.
        #                 for (v1, v2) in combinations(new_simplex, 2):
        #                     if v1==v2 or (v1, v2) in self._simplices[1] or (v2, v1) in self._simplices[1]:
        #                         continue
        #                     new_dist = np.linalg.norm(self.coordinates[v1]-self.coordinates[v2])
        #                     if new_dist > max_dist:
        #                         max_dist = new_dist
        #                         v_max = [v1, v2]
        #                     # if new_dist < min_dist:
        #                     #     min_dist = new_dist
        #                     #     v_max = [v1, v2]
        #                 # print(v_max)
        #                 # v_dists = cdist(v_coords, v_coords)
        #                 # v_max = np.unravel_index(np.argmax(v_dists), v_dists.shape)
        #                 # dividing simplex
        #                 div_simplex = list(set(new_simplex).difference(v_max))
        #                 comp_simplex_1 = list(set(new_simplex).difference([v_max[0]]))
        #                 comp_simplex_2 = list(set(new_simplex).difference([v_max[1]]))
        #                 div_simplex.sort()
        #                 comp_simplex_1.sort()
        #                 comp_simplex_2.sort()
        #                 self._simplices[self.dimension-1].append(tuple(div_simplex))
        #                 self._simplices[self.dimension].append(tuple(comp_simplex_1))
        #                 self._simplices[self.dimension].append(tuple(comp_simplex_2))

    # ### PATCHING
    # if patching:
    #     # d+1 dimension
        #     extra_simplices = []
        #     for x_eps_lms in eps_lms:
        #         if len(x_eps_lms) >= self.dimension + 2:
        #             new_simplex = tuple(np.sort(x_eps_lms[:self.dimension + 2]))
        #             if new_simplex in extra_simplices:
        #                 continue
        #             # # check if all boundary elements are already present
        #             # bdQ = True
        #             # for b in range(d + 1):
        #             #     boundary_simplex = new_simplex[:b] + new_simplex[b + 1:]
        #             #     if boundary_simplex not in self._simplices[d - 1]:
        #             #         bdQ = False
        #             #         break
        #             # if bdQ and (new_simplex not in self._simplices[d]):
        #             #     self._simplices[d].append(new_simplex)
        #
        #             # 3-d case
        #             bdQ = True
        #             count_codim2 = dict()
        #             for s in combinations(new_simplex, self.dimension-1):
        #                 count_codim2[s] = 0
        #
        #             for boundary_simplex in combinations(new_simplex, self.dimension):
        #                 if boundary_simplex in self.simplices[self.dimension-1]:
        #                     for bbs in combinations(boundary_simplex, self.dimension-1):
        #                         count_codim2[bbs] += 1
        #             if all([x == 2 for x in count_codim2.values()]):
        #                 extra_simplices.append(tuple(new_simplex))
        #                 # find two the most distant vertices:
        #                 v_coords = np.array([self.coordinates[i] for i in new_simplex])
        #
        #                 v_max = []
        #                 max_dist = -1
        #                 min_dist = 10000000.
        #                 for (v1, v2) in combinations(new_simplex, 2):
        #                     if v1==v2 or (v1, v2) in self._simplices[1] or (v2, v1) in self._simplices[1]:
        #                         continue
        #                     new_dist = np.linalg.norm(self.coordinates[v1]-self.coordinates[v2])
        #                     if new_dist > max_dist:
        #                         max_dist = new_dist
        #                         v_max = [v1, v2]
        #                     # if new_dist < min_dist:
        #                     #     min_dist = new_dist
        #                     #     v_max = [v1, v2]
        #                 # print(v_max)
        #                 # v_dists = cdist(v_coords, v_coords)
        #                 # v_max = np.unravel_index(np.argmax(v_dists), v_dists.shape)
        #                 # dividing simplex
        #                 div_simplex = list(set(new_simplex).difference(v_max))
        #                 comp_simplex_1 = list(set(new_simplex).difference([v_max[0]]))
        #                 comp_simplex_2 = list(set(new_simplex).difference([v_max[1]]))
        #                 div_simplex.sort()
        #                 comp_simplex_1.sort()
        #                 comp_simplex_2.sort()
        #                 self._simplices[self.dimension-1].append(tuple(div_simplex))
        #                 self._simplices[self.dimension].append(tuple(comp_simplex_1))
        #                 self._simplices[self.dimension].append(tuple(comp_simplex_2))
        #
        #                 # print(div_simplex, comp_simplex_1, comp_simplex_2)
        #                 # print(v_dists)
        #                 # print(v_max)
        #
        #                 # print(count_codim2)

        # for v in self.simplices[0]:
        #     print(v, ' - ', self.coordinates[v[0]])
        # self._st.persistence()