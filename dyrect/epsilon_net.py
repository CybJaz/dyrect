from gudhi import SimplexTree
from itertools import combinations
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.distance import cdist

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

    @property
    def complex(self):
        return self._complex


class NerveComplex:
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
                simplex = tuple(np.where(distances[i, :] <= eps)[0])
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
    def epsilon(self):
        return self._eps