import functools
import itertools
import sys
from typing import List, Any

import numpy as np
import networkx as nx
from gudhi import SimplexTree
from itertools import chain, combinations
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.distance import cdist

from .complex import Complex
from .cgal_utils import in_convex_hull

class WitnessComplex(Complex):
    def __init__(self, landmarks, witnesses, max_dim, beta_sq=0.0, alpha=-1):
        """
        :param lms:
        :param alpha:
        :param max_dim:
        :param points:
        """
        """
            TODO: if list of points is empty then generate just a rips-complex
            TODO: adjust code for max_dim=-1
        """
        self._st = SimplexTree()
        self._simplices = dict()
        # self._coordinates = landmarks
        self._alpha = alpha
        self._beta_square = beta_sq
        self._dim = max_dim
        # """ ambient space dimension """
        # self._ambient_dim = landmarks.shape[1]

        if type(landmarks) == dict:
            self._coordinates = landmarks
            self._ambient_dim = len(list(landmarks.values()[0]))
        else:
            self._ambient_dim = landmarks.shape[1]
            self._coordinates = {idx: coord for idx, coord in enumerate(landmarks)}

        self._barren_witnesses = dict()
        self._weakly_witnessed = dict()

        if alpha > 0:
            distances = cdist(witnesses, np.array(list(self._coordinates.values())), 'euclidean')
            argsort_dists = np.argsort(distances, axis=1)
            distances.sort(axis=1)
        else:
            argsort_dists = self._argsort_distances(witnesses)

        if self._alpha > 0 and self._beta_square >= 0.:
            ### 0-simplices
            self._simplices[0] = []
            for i in range(len(witnesses)):
                if distances[i, 0] <= self._alpha:
                    simplex = (argsort_dists[i, 0],)
                    if simplex not in self._simplices[0]:
                        self._simplices[0].append(simplex)

            for d in range(1, max_dim + 1):
                self._simplices[d] = []
                self._barren_witnesses[d] = []
                self._weakly_witnessed[d] = []
                for i in range(len(witnesses)):
                    dl = d
                    while distances[i, dl+1]**2 <= distances[i, d]**2 + self._beta_square:
                        dl += 1
                    beta_simplex = argsort_dists[i, :(dl+1)]
                    # beta_simplex.sort()
                    for d_beta_simplex in combinations(beta_simplex, d+1):
                        max_in = 0.
                        min_out = distances[i, dl]

                        for iv, v in enumerate(beta_simplex):
                            if v in d_beta_simplex:
                                max_in = max(max_in, distances[i, iv])
                            else:
                                min_out = min(min_out, distances[i, iv])

                        if max_in**2 <= min_out**2 + self._beta_square:
                            simplex = tuple(np.sort(d_beta_simplex))
                            if simplex not in self._simplices[d]:
                                well_witnessed = True
                                for face in combinations(simplex, d):
                                    if face not in self._simplices[d - 1]:
                                        well_witnessed = False
                                        break
                                if well_witnessed:
                                    self._simplices[d].append(simplex)
                                else:
                                    self._barren_witnesses[d].append(i)
                                    if simplex not in self._weakly_witnessed[d]:
                                        self._weakly_witnessed[d].append(simplex)

        elif self._alpha > 0:
            # assert False, "an option with positive epsilon needs to be tested"
            ### 0-simplices
            self._simplices[0] = []
            for i in range(len(witnesses)):
                if distances[i, 0] <= self._alpha:
                    simplex = (argsort_dists[i, 0],)
                    if simplex not in self._simplices[0]:
                        self._simplices[0].append(simplex)

            for d in range(1, max_dim + 1):
                self._simplices[d] = []
                self._barren_witnesses[d] = []
                self._weakly_witnessed[d] = []
                for i in range(len(witnesses)):
                    if distances[i, d] <= self._alpha:
                        simplex = tuple(np.sort(argsort_dists[i, :d + 1]))
                        if simplex not in self._simplices[d]:
                            well_witnessed = True
                            for face in combinations(simplex, d):
                                if face not in self._simplices[d - 1]:
                                    well_witnessed = False
                                    break
                            if well_witnessed:
                                self._simplices[d].append(simplex)
                            else:
                                self._barren_witnesses[d].append(i)
                                if simplex not in self._weakly_witnessed[d]:
                                    self._weakly_witnessed[d].append(simplex)
        else:
            ### 0-simplices
            self._simplices[0] = []
            for i in range(len(witnesses)):
                # if distances[i, argsort_dists[i, 0]] <= self._eps:
                simplex = (argsort_dists[i, 0],)
                if simplex not in self._simplices[0]:
                    self._simplices[0].append(simplex)
            print("Dimension ", 0, " completed")

            for d in range(1, max_dim + 1):
                self._simplices[d] = []
                self._barren_witnesses[d] = []
                # self._weakly_witnessed[d] = []
                for i in range(len(witnesses)):
                    # if distances[i, argsort_dists[i, d]] <= self._eps:
                    simplex = tuple(np.sort(argsort_dists[i, :d + 1]))
                    if simplex not in self._simplices[d]:
                        well_witnessed = True
                        for face in combinations(simplex, d):
                            if face not in self._simplices[d - 1]:
                                well_witnessed = False
                                break
                        if well_witnessed:
                            self._simplices[d].append(simplex)
                            self._st.insert(list(simplex))
                        else:
                            self._barren_witnesses[d].append(i)
                            # if simplex not in self._weakly_witnessed[d]:
                            #     self._weakly_witnessed[d].append(simplex)
                print("Dimension ", d, " completed")
        print("witness constructed")
        self._st.compute_persistence(persistence_dim_max=True)
        print("betti computed")
        self._betti_numbers = self._st.betti_numbers()

    @classmethod
    def construct(cls, st, simpls, coords, dim):
        obj = cls.__new__(cls)
        obj._st = st
        obj._simplices = simpls
        if type(coords) == dict:
            obj._coordinates = coords
            obj._ambient_dim = len(list(coords.values()[0]))
        else:
            obj._ambient_dim = coords.shape[1]
            obj._coordinates = {idx: coord for idx, coord in enumerate(coords)}
        obj._dim = dim
        obj._betti_numbers = None
        return obj

    def _argsort_distances(self, points):
        skip = 50000
        argsort_dists = np.zeros((len(points), len(self._coordinates.values())), dtype=np.int32)
        for p in np.arange(0, len(points), skip):
            distances = cdist(points[p:(p+skip)], np.array(list(self._coordinates.values())), 'euclidean')
            argsort_dists_slice = np.argsort(distances, axis=1)
            argsort_dists[p:(p+skip), :] = argsort_dists_slice

        # distances = cdist(points, np.array(list(self._coordinates.values())), 'euclidean')
        # argsort_dists = np.argsort(distances, axis=1)
        return argsort_dists

    def barren_witnesses(self, points, dim, indices=False):
        barrens = []
        ibarrens = []
        argsort_dists = self._argsort_distances(points)
        for ip, p in enumerate(points):
            wsim = argsort_dists[ip, :dim+1]
            wsim.sort()
            if tuple(wsim) not in self._simplices[dim]:
                barrens.append(p)
                ibarrens.append(ip)
        if indices:
            return np.array(ibarrens)
        else:
            return np.array(barrens)


class EdgeCliqueWitnessComplex(WitnessComplex):
    def __init__(self, landmarks, witnesses, witnesses_dim, max_cliques_dim=100, max_complex_dimension=-1):
        """
        @param landmarks:
        @param witnesses:
        @param max_dim:
        @param all_cliques: if True, include all cliques as simplices, otherwise ignore cliques correspondig to higher
                dimensional simplices
        """
        self._st = SimplexTree()
        self._simplices = dict()
        if max_complex_dimension >= 0:
            self._dim = max_complex_dimension
        else:
            self._dim = witnesses_dim
        self._betti_numbers = None

        if type(landmarks) == dict:
            self._coordinates = landmarks
            self._ambient_dim = len(list(landmarks.values()[0]))
        else:
            self._ambient_dim = landmarks.shape[1]
            self._coordinates = {idx: coord for idx, coord in enumerate(landmarks)}

        print("simplices: ", np.sum([len(d) for d in self._simplices.values()]))
        argsort_dists = self._argsort_distances(witnesses)

        for d in range(self._dim+1):
            self._simplices[d] = []
        for i in range(len(witnesses)):
            simplex = (argsort_dists[i, 0],)
            if simplex not in self._simplices[0]:
                self._simplices[0].append(simplex)
                self._st.insert(list(simplex))

        self._vmatrix = VMatrix(len(self._coordinates))
        for iw in range(len(witnesses)):
            # for d in range(1, self._dim+1):
            dsim = argsort_dists[iw, :(witnesses_dim+1)]
            self._vmatrix.add_directed_simplex(dsim)
        g = nx.from_numpy_matrix(self._vmatrix.uni_vmatrix)
        for clique in nx.find_cliques(g):
            ds = len(clique) - 1
            clique.sort()
            if ds > 0 and ds <= max_cliques_dim:
                if ds not in self._simplices:
                    self._simplices[ds] = []
                if max_complex_dimension < 0:
                    dim_range = range(1, ds+1)
                else:
                    dim_range = range(1, self._dim+1)
                # self._simplices[ds].append(tuple(clique))
                for fd in dim_range:
                    for face in itertools.combinations(clique, fd+1):
                        fsim = tuple(face)
                        if fsim not in self._simplices[fd]:
                            self._simplices[fd].append(fsim)
                            self._st.insert(list(fsim))
            # elif ds > self._dim:

    def barrens_patching(self, points, dim, level=1, over=0):
        barren_witnesses = self.barren_witnesses(points, dim)
        argsort_dists = self._argsort_distances(barren_witnesses)

        self._vmatrix = VMatrix(len(self._coordinates))
        for iw in range(len(barren_witnesses)):
            dsim = argsort_dists[iw, :(self._dim+1+level)]
            self._vmatrix.add_directed_simplex(dsim)
        g = nx.from_numpy_matrix(self._vmatrix.uni_vmatrix)
        for clique in nx.find_cliques(g):
            ds = len(clique) - 1
            clique.sort()
            if ds not in self._simplices:
                nd = ds
                while nd not in self._simplices:
                    self._simplices[nd] = []
                    nd -= 1
            if ds >= dim:
                # print(clique)
                # if ds <= self._dim:
                #     self._simplices[ds].append(tuple(clique))
                for fd in range(1, ds+1):
                    for face in itertools.combinations(clique, fd+1):
                        fsim = tuple(face)
                        if fsim not in self._simplices[fd]:
                            self._simplices[fd].append(fsim)
        self._st = None
        self._betti_numbers = None

    def voted_barrens_patching(self, points, dim, level=1, over=0):
        barren_witnesses = self.barren_witnesses(points, dim)
        argsort_dists = self._argsort_distances(barren_witnesses)

        self._vmatrix = VMatrix(len(self._coordinates))
        for iw in range(len(barren_witnesses)):
            dsim = argsort_dists[iw, :(self._dim+1+level)]
            self._vmatrix.add_directed_simplex(dsim)
        univ = self._vmatrix.uni_vmatrix
        g = nx.from_numpy_matrix(univ)

        for clique in nx.find_cliques(g):
            ds = len(clique) - 1
            clique.sort()
            if ds >= dim:
                for fd in range(ds, dim-1, -1):
                    for face in itertools.combinations(clique, fd+1):
                        # print(face)
                        subcomplex = self.subcomplex(face)
                        if is_simple_cycle(subcomplex, dim-1):
                            new_simplices = []
                            weights = []
                            for edge in list(itertools.combinations(face, dim)):
                                if tuple(edge) not in self._simplices[dim-1]:
                                    new_simplices.append(tuple(edge))
                                    weights.append(univ[edge[0], edge[1]])
                            if len(weights) == 0:
                                self.add_clique(face)
                            else:
                                medge = np.argmax(weights)
                                self.add_clique(new_simplices[medge])

            # for fd in range(1, ds+1):
                #     for face in itertools.combinations(clique, fd+1):
                #         fsim = tuple(face)
                #         if fsim not in self._simplices[fd]:
                #             self._simplices[fd].append(fsim)
        self._betti_numbers = None

    def add_clique(self, clique):
        csim = np.sort(clique)
        clen = len(clique)
        if clen-1 not in self._simplices:
            self._simplices[clen-1] = []
        for fd in range(clen):
            for face in itertools.combinations(clique, fd + 1):
                fsim = tuple(face)
                if fsim not in self._simplices[fd]:
                    self._simplices[fd].append(fsim)
                    self._st.insert(list(fsim))
                    # print(fsim)


def is_simple_cycle(complex, dim):
    if (dim+1) in complex.simplices and len(complex.simplices[dim+1]) > 0:
        return False

    num_of_cofaces = {i: 0 for i in complex._simplices[dim-1]}
    for s in complex._simplices[dim]:
        for face in combinations(s, dim):
            num_of_cofaces[face] += 1

    if all(np.array(list(num_of_cofaces.values())) == 2):
        return True
    else:
        return False


class VMatrix():
    def __init__(self, n):
        self._n = n
        i16 = np.iinfo(np.int16)
        # self._vmatrix = np.diag([i16.max for _ in range(self._n)])
        self._vmatrix = np.zeros((self._n, self._n))

    def add_directed_simplex(self, dsimplex):
        """
        increase counter for directed edges of the form (dsimplex[0], dsimplex[v]), v>0
        @param dsimplex: a directed simplex, aka. a sorted tuple/array
        @return:
        """
        v0 = dsimplex[0]
        for v in dsimplex[1:]:
            self._vmatrix[v0, v] += 1

    def at(self, x, y):
        return self._vmatrix[x, y]

    @property
    def uni_vmatrix(self):
        umatrix = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
                if self._vmatrix[i, j] != 0 and self._vmatrix[j, i] != 0:
                    v = min(self._vmatrix[i, j], self._vmatrix[j, i])
                    umatrix[i, j] = v
                    umatrix[j, i] = v
        return umatrix


class EdgeCliqueSimplex():
    def __init__(self, simplex):
        self._n = len(simplex)
        self._dim = self._n - 1
        self._simplex = tuple(np.sort(simplex))
        self._dedges = {de: 0 for de in itertools.permutations(simplex, 2)}
        i16 = np.iinfo(np.int16)
        self._vmatrix = np.diag([i16.max for _ in range(self._n)])

    @property
    def simplex(self):
        return self._simplex

    def add_directed_simplex(self, dsimplex):
        """
        increase counter for directed edges of the form (dsimplex[0], dsimplex[v]), v>0
        @param dsimplex: a directed simplex, aka. a sorted tuple/array
        @return:
        """
        v0 = dsimplex[0]
        for v in dsimplex[1:]:
            self._dedges[(v0, v)] += 1

    def is_clique(self):
        if np.min(list(self._dedges.values())) > 0:
            return True
        else:
            return False



class VWitnessComplex(WitnessComplex):
    def __init__(self, landmarks, witnesses, max_dim, alpha=-1, v=[0]):
        """

        @param landmarks:
        @param witnesses:
        @param max_dim:
        @param alpha:
        @param v: len of this array should match max_dim
        """
        assert max_dim + 1 == len(v)
        # super(VWitnessComplex, self).__init__(landmarks, witnesses, max_dim, alpha, v)
        self._st = SimplexTree()
        self._simplices = dict()
        # self._coordinates = landmarks
        self._alpha = alpha
        self._dim = max_dim

        if type(landmarks) == dict:
            self._coordinates = landmarks
            self._ambient_dim = len(list(landmarks.values()[0]))
        else:
            self._ambient_dim = landmarks.shape[1]
            self._coordinates = {idx: coord for idx, coord in enumerate(landmarks)}

        distances = cdist(witnesses, np.array(list(self._coordinates.values())), 'euclidean')
        argsort_dists = np.argsort(distances, axis=1)
        distances.sort(axis=1)

        self._patched = dict()


        ### 0-simplices
        if alpha > 0:
            self._simplices[0] = []
            for i in range(len(witnesses)):
                if distances[i, 0] <= self._alpha:
                    simplex = (argsort_dists[i, 0],)
                    if simplex not in self._simplices[0]:
                        self._simplices[0].append(simplex)

            for vi, vv in enumerate(v[1:]):
            # for d in range(1, max_dim + 1):
                d = vi + 1
                self._simplices[d] = []
                for i in range(len(witnesses)):
                    maxd = d
                    while distances[i, d] > self._alpha:
                        maxd -= 1

                    neighbors = tuple(np.sort(argsort_dists[i, :(maxd+vv+1)]))
                    for simplex in combinations(neighbors, d + 1):
                        simplex = tuple(np.sort(simplex))
                        well_witnessed = True
                        for face in combinations(simplex, d):
                            if face not in self._simplices[d - 1]:
                                well_witnessed = False
                                break
                        if well_witnessed and simplex not in self._simplices[d]:
                            self._simplices[d].append(simplex)
        else:
            self._simplices[0] = []
            for i in range(len(witnesses)):
                if distances[i, 0] <= self._alpha:
                    simplex = (argsort_dists[i, 0],)
                    if simplex not in self._simplices[0]:
                        self._simplices[0].append(simplex)

            for vi, vv in enumerate(v[1:]):
                # for d in range(1, max_dim + 1):
                d = vi + 1
                self._simplices[d] = []
                for i in range(len(witnesses)):
                    maxd = d
                    neighbors = tuple(np.sort(argsort_dists[i, :(maxd+vv+1)]))
                    for simplex in combinations(neighbors, d + 1):
                        simplex = tuple(np.sort(simplex))
                        well_witnessed = True
                        for face in combinations(simplex, d):
                            if face not in self._simplices[d - 1]:
                                well_witnessed = False
                                break
                        if well_witnessed and simplex not in self._simplices[d]:
                            self._simplices[d].append(simplex)



class PatchedWitnessComplex(WitnessComplex):
    def __init__(self, landmarks, witnesses, max_dim, alpha=-1,
                 patching_level=1, max_patched_dimensions=2, patching_type="knitting"):
        super(PatchedWitnessComplex, self).__init__(landmarks, witnesses, max_dim, alpha)

        # print("simplex tree: ", sys.getsizeof(self._st.copy))
        print("simplices: ", np.sum([len(d) for d in self._simplices.values()]))
        # distances = cdist(witnesses, np.array(list(self._coordinates.values())), 'euclidean')
        # argsort_dists = np.argsort(distances, axis=1)
        # distances.sort(axis=1)
        argsort_dists = self._argsort_distances(witnesses)

        self._patched = dict()

        # TODO: incorporate eps into the patching
        if alpha > 0:
            print("WARNING: eps is not taken into the account during the patching yet")

        count_continued = 0
        # Patched dimension
        # e.g. pd=2 is for closing 1-dim holes
        # for pd in range(2, max_patched_dimensions + 1):
        for pd in range(max_patched_dimensions, 1, -1):
            print("Patching dimension: ", pd)
            self._patched[pd] = []
            # e.g. cpd=1 uses 3-simplices for closing 1-holes
            # e.g. cpd=2 uses 4-simplices for closing 1-holes
            for cpd in range(1, patching_level + 1):
                print("patched list: ", np.sum([len(d) for d in self._patched.values()]))
            # for cpd in range(patching_level, patching_level + 1):
                ### Number of vertices of the patching simplex
                nvp = pd + cpd + 1
                # extra_simplices = []

                if pd not in self._barren_witnesses:
                    bwitnesses = range(len(witnesses))
                else:
                    bwitnesses = self._barren_witnesses[pd]
                print("Checking ", len(bwitnesses), " barren witnesses")
                for x in bwitnesses:
                    new_simplex = tuple(np.sort(argsort_dists[x, :nvp]))
                    if new_simplex in self._patched[pd]:
                        count_continued += 1
                        continue
                    # if new_simplex in extra_simplices:
                    #     continue

                    # TODO: first gather the list of all potential cycles than start reducing it
                    # this way there will be much less cycle reductions
                    hole_subcomplex = self.subcomplex(new_simplex)
                    self.cycle_reduction(hole_subcomplex)

                    new_reduced_simplex = tuple(hole_subcomplex.vertices)
                    if new_reduced_simplex in self._patched[pd] or len(new_reduced_simplex) == 1:
                        count_continued += 1
                        continue

                    self._patched[pd].append(new_reduced_simplex)

                    ### Check if new_simplex is a simple cycle
                    if len(hole_subcomplex._simplices[pd]) == 0:
                        num_of_cofaces = {i: 0 for i in hole_subcomplex._simplices[pd - 2]}
                        for s in hole_subcomplex._simplices[pd - 1]:
                            for face in combinations(s, pd - 1):
                                num_of_cofaces[face] += 1
                        if all(np.array(list(num_of_cofaces.values())) == 2):
                            # print(hole_subcomplex.betti_numbers)
                            # self.knitting_patching(hole_subcomplex)
                            # self.web_patching(hole_subcomplex)
                            self.fan_patching(hole_subcomplex)
                            # self._patched[pd].append(new_simplex)
                            # self.morse_patching(hole_subcomplex)
                            self.merge_complex(hole_subcomplex)

        print("simplices: ", np.sum([len(d) for d in self._simplices.values()]))
        print("Continued: " + str(count_continued))

    def _simplex_distance(self, c, sim):
        b = np.mean(self.coords_list(list(sim)), axis=0)
        return np.linalg.norm(c - b)

    def cycle_reduction(self, hole_subcomplex):
        reduced = False
        while not reduced:
            c = np.mean(self.coords_list([v for (v,) in hole_subcomplex.simplices[0]]), axis=0)
            reduced = True
            reducible_pairs = []
            weights = []
            for (sim, _) in hole_subcomplex._st.get_simplices():
                cofaces = hole_subcomplex._st.get_cofaces(sim, 1)
                if len(cofaces)==1:
                    reducible_pairs.append([sim, cofaces[0][0]])
                    weights.append(self._simplex_distance(c, sim))
            if len(reducible_pairs) > 0:
                ri = np.argmax(weights)
                hole_subcomplex.remove_simplex(reducible_pairs[ri][1])
                hole_subcomplex.remove_simplex(reducible_pairs[ri][0])
                reduced = False

    def fan_patching(self, hole_subcomplex):
        vertices = [v for (v,) in hole_subcomplex.simplices[0]]
        b = np.mean(self.coords_list(vertices), axis=0)
        weights = [np.linalg.norm(b - self.coordinates[v]) for v in vertices]
        v0 = vertices[np.argmin(weights)]

        dims = list(hole_subcomplex.simplices.keys())
        dims.sort(reverse=True)
        hole_subcomplex.simplices[dims[0]+1] = []
        for d in dims:
            for s in hole_subcomplex.simplices[d]:
                if v0 in s:
                    continue
                new_simplex = list(s) + [v0]
                new_simplex.sort()
                new_simplex = tuple(new_simplex)
                if new_simplex not in hole_subcomplex.simplices[d+1]:
                    hole_subcomplex.simplices[d+1].append(new_simplex)
        hole_subcomplex.simplices[0].append((v0,))


    # def fan_patching(self, hole_subcomplex):
    #     vertices = [v for (v,) in hole_subcomplex.simplices[0]]
    #     vertices.sort()
    #
    #     # if hole is not convex choose a point in the interior of the convex hull
    #     v0 = in_convex_hull([self.coordinates[v] for v in vertices])
    #     if v0 is not None:
    #         v0 = vertices[v0]
    #     else:
    #         v0 = vertices[0]
    #
    #     dims = list(hole_subcomplex.simplices.keys())
    #     dims.sort(reverse=True)
    #     hole_subcomplex.simplices[dims[0]+1] = []
    #     for d in dims:
    #         for s in hole_subcomplex.simplices[d]:
    #             if v0 in s:
    #                 continue
    #             new_simplex = list(s) + [v0]
    #             new_simplex.sort()
    #             new_simplex = tuple(new_simplex)
    #             if new_simplex not in hole_subcomplex.simplices[d+1]:
    #                 hole_subcomplex.simplices[d+1].append(new_simplex)
    #     hole_subcomplex.simplices[0].append((v0,))


    ### UNFINISHED METHOD!
    def morse_patching(self, hole_subcomplex):
        """
        @param hole_subcomplex:
        @return:
        """
        vertices = [v for (v,) in hole_subcomplex.simplices[0]]
        vertices.sort()

        rigid_simplices = set()
        for d in hole_subcomplex.simplices:
            for sim in hole_subcomplex.simplices[d]:
                rigid_simplices.add(sim)

        for d in range(2, len(vertices)+1):
            for sim in combinations(vertices, d):
                sim = tuple(sim)
                if d-1 not in hole_subcomplex.simplices or sim not in hole_subcomplex.simplices[d-1]:
                    hole_subcomplex.add_simplex(sim)

#        patch_simplices = {d: [tuple(np.sort(sim)) for sim in combinations(vertices, d)]
#                           for d in range(1, len(vertices))}
#        patch_complex = Complex(patch_simplices)

        # center
        center = np.mean(self.coords_list([v for (v,) in hole_subcomplex.simplices[0]]), axis=0)
        vdists = {v: np.linalg.norm(center - self.coordinates[v]) for v in vertices}
        weights = {}
        for d in hole_subcomplex.simplices:
            for sim in hole_subcomplex.simplices[d]:
                # weights[sim] = self._simplex_distance(center, sim)
                weights[sim] = np.max([vdists[x] for x in sim])


        maxd = len(vertices) - 2
        reduced = False
        while not reduced:
            reduced = True
            reducible_pairs = []
            w = []
            for d in range(maxd, 0, -1):
                if len(hole_subcomplex.simplices[d+1]) == 0:
                    continue
                for sim in hole_subcomplex.simplices[d]:
                # for (sim, _) in hole_subcomplex._st.get_simplices():
                #     print(sim)
                    if sim not in rigid_simplices:
                        cofaces = hole_subcomplex._st.get_cofaces(sim, 1)
                        if len(cofaces) == 1:
                            coface = tuple(cofaces[0][0])
                            if coface not in rigid_simplices:
                                reducible_pairs.append([sim, coface])
                                w.append(weights[sim])
                if len(reducible_pairs) > 0:
                    break
            if len(reducible_pairs) > 0:
                ri = np.argmax(w)
                # print("reducing: ", reducible_pairs[ri])
                hole_subcomplex.remove_simplex(reducible_pairs[ri][1])
                hole_subcomplex.remove_simplex(reducible_pairs[ri][0])
                reduced = False
        # print(hole_subcomplex.betti_numbers)


    def knitting_patching(self, hole_subcomplex):
        vertices = [v for (v,) in hole_subcomplex.simplices[0]]
        vertices.sort()
        if len(vertices) == 6:
            print(6)
        potential_edges = [tuple(e) for e in combinations(vertices, 2)]
        for e in hole_subcomplex.simplices[1]:
            potential_edges.remove(e)

        # potential_edges = [tuple(e) for e in combinations(vertices, 2) if tuple(e) not in hole_subcomplex.simplices[1]]
        edges_lengths = [np.linalg.norm(self.coordinates[e1] - self.coordinates[e2]) for (e1, e2) in potential_edges]
        argsorted_edges = list(np.argsort(edges_lengths))
        while len(vertices) > 0:
            eidx = argsorted_edges.pop(0)
            (v1, v2) = potential_edges[eidx]
            if v1 not in vertices or v2 not in vertices:
                continue
            vertices.remove(v1)
            vertices.remove(v2)
            new_edge = tuple(np.sort([v1, v2]))
            hole_subcomplex.add_simplex(new_edge)
            # hole_subcomplex.simplices[1].append(new_edge)

            triangle_exists = False
            for i in reversed(range(len(vertices))):
                tv = vertices[i]
                e1 = tuple(np.sort([v1, tv]))
                e2 = tuple(np.sort([v2, tv]))
                if e1 in hole_subcomplex.simplices[1] and e2 in hole_subcomplex.simplices[1]:
                    new_triangle = tuple(np.sort([v1, v2, tv]))
                    # hole_subcomplex.simplices[2].append(new_triangle)
                    hole_subcomplex.add_simplex(new_triangle)
                    vertices.remove(tv)
                    triangle_exists = True

            if triangle_exists and len(vertices) > 1:
                vertices.append(v1)
                vertices.append(v2)
            elif not triangle_exists:
                print("todo")
                for v in vertices:
                    if tuple(np.sort([v, v1])) in hole_subcomplex.simplices[1]:
                        sv1 = v
                        vertices.remove(sv1)
                        break
                simple_subhole_1 = [sv1]
                found_next = True
                sv = simple_subhole_1[-1]
                while found_next:
                    found_next = False
                    for v in vertices:
                        if tuple(np.sort([sv, v])) in hole_subcomplex.simplices[1]:
                            simple_subhole_1.append(v)
                            sv = v
                            vertices.remove(v)
                            found_next = True
                            break
                simple_subhole_1 = simple_subhole_1 + [v1, v2]
                simple_subhole_2 = vertices + [v1, v2]
                vertices = []
                subproblem1 = hole_subcomplex.subcomplex(simple_subhole_1)
                subproblem2 = hole_subcomplex.subcomplex(simple_subhole_2)
                self.knitting_patching(subproblem1)
                hole_subcomplex.merge_complex(subproblem1)
                self.knitting_patching(subproblem2)
                hole_subcomplex.merge_complex(subproblem2)
