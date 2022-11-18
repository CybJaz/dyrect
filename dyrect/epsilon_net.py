import itertools

from gudhi import SimplexTree
from itertools import combinations
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.distance import cdist

from .poset import Poset
from itertools import chain, combinations

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
    def __init__(self, max_dim=-1):
        self._simplices = dict()
        self._coordinates = None
        self._dim = max_dim

    @classmethod
    def construct(cls, simpls, coords=None, max_dim = -1):
        obj = cls.__new__(cls)
        obj._simplices = simpls
        obj._coordinates = coords
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

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def simplices(self):
        return self._simplices

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


class AlphaNerveComplex(NerveComplex):
    def __init__(self, lms, eps, max_dim=-1, points=[], patching=True, record_witnesses=False):
        """
        :param lms:
        :param eps:
        :param max_dim:
        :param points:
        """
        # NerveComplex.__init__(self, lms, eps, max_dim, points=points)

        self._st = SimplexTree()
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

        # for each point get a sorted (increasing) list of landmarks no further than epsilon times coefficient
        eps_lms = []
        for x_arg_sorted, x in zip(np.argsort(distances, axis=1), range(num_of_points)):
            i = 0
            while distances[x, x_arg_sorted[i]] < eps * 2.3:
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

        # 1 to d-dimension;
        # for dimension 2 - add a 2-simplex (a,b) if there exists a point x such that the list of
        #   the closest landmarks starts either with (a,b,...) or (b,a,...)
        # for dimension 3 - add a 3-simplex (a,b,c) if there exists a point x such that the list of
        #   the closest landmarks starts with (a,b,c), e.g. (b,a,c,d,f,e,...),
        #   and all faces of (a,b,c) are already in the complex
        for d in range(1, self.dimension + 1):
            # print("Dimension: ", d)
            for x_eps_lms in eps_lms:
                if len(x_eps_lms) >= d+1:
                    new_simplex = tuple(np.sort(x_eps_lms[:d+1]))
                    # check if all boundary elements are already present
                    bdQ = True
                    for b in range(d+1):
                        boundary_simplex = new_simplex[:b] + new_simplex[b+1:]
                        if boundary_simplex not in self._simplices[d-1]:
                            bdQ = False
                            break
                    if bdQ and (new_simplex not in self._simplices[d]):
                        self._simplices[d].append(new_simplex)

        if record_witnesses:
            self.non_witnesses = {0: [], 1: [], 2: []}
            self.not_witnessed = {0: [], 1: [], 2: []}
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

        ### PATCHING
        if patching:
            # d+1 dimension
            extra_simplices = []
            for x_eps_lms in eps_lms:
                if len(x_eps_lms) >= self.dimension + 2:
                    new_simplex = tuple(np.sort(x_eps_lms[:self.dimension + 2]))
                    if new_simplex in extra_simplices:
                        continue
                    # # check if all boundary elements are already present
                    # bdQ = True
                    # for b in range(d + 1):
                    #     boundary_simplex = new_simplex[:b] + new_simplex[b + 1:]
                    #     if boundary_simplex not in self._simplices[d - 1]:
                    #         bdQ = False
                    #         break
                    # if bdQ and (new_simplex not in self._simplices[d]):
                    #     self._simplices[d].append(new_simplex)

                    # 3-d case
                    bdQ = True
                    count_codim2 = dict()
                    for s in combinations(new_simplex, self.dimension-1):
                        count_codim2[s] = 0

                    for boundary_simplex in combinations(new_simplex, self.dimension):
                        if boundary_simplex in self.simplices[self.dimension-1]:
                            for bbs in combinations(boundary_simplex, self.dimension-1):
                                count_codim2[bbs] += 1
                    if all([x == 2 for x in count_codim2.values()]):
                        extra_simplices.append(tuple(new_simplex))
                        # find two the most distant vertices:
                        v_coords = np.array([self.coordinates[i] for i in new_simplex])

                        v_max = []
                        max_dist = -1
                        min_dist = 10000000.
                        for (v1, v2) in combinations(new_simplex, 2):
                            if v1==v2 or (v1, v2) in self._simplices[1] or (v2, v1) in self._simplices[1]:
                                continue
                            new_dist = np.linalg.norm(self.coordinates[v1]-self.coordinates[v2])
                            if new_dist > max_dist:
                                max_dist = new_dist
                                v_max = [v1, v2]
                            # if new_dist < min_dist:
                            #     min_dist = new_dist
                            #     v_max = [v1, v2]
                        # print(v_max)
                        # v_dists = cdist(v_coords, v_coords)
                        # v_max = np.unravel_index(np.argmax(v_dists), v_dists.shape)
                        # dividing simplex
                        div_simplex = list(set(new_simplex).difference(v_max))
                        comp_simplex_1 = list(set(new_simplex).difference([v_max[0]]))
                        comp_simplex_2 = list(set(new_simplex).difference([v_max[1]]))
                        div_simplex.sort()
                        comp_simplex_1.sort()
                        comp_simplex_2.sort()
                        self._simplices[self.dimension-1].append(tuple(div_simplex))
                        self._simplices[self.dimension].append(tuple(comp_simplex_1))
                        self._simplices[self.dimension].append(tuple(comp_simplex_2))

                        # print(div_simplex, comp_simplex_1, comp_simplex_2)
                        # print(v_dists)
                        # print(v_max)

                        # print(count_codim2)

        # for v in self.simplices[0]:
        #     print(v, ' - ', self.coordinates[v[0]])
        # self._st.persistence()