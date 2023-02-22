import numpy as np
from gudhi import SimplexTree
from itertools import chain, combinations
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.distance import cdist

from .epsilon_net import Complex

class WitnessComplex(Complex):
    def __init__(self, landmarks, witnesses,  max_dim, eps=-1):
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
        self._coordinates = landmarks
        self._eps = eps
        self._dim = max_dim
        """ ambient space dimension """
        self._ambient_dim = landmarks.shape[1]

        self._barren_witnesses = dict()
        self._weakly_witnessed = dict()

        distances = cdist(witnesses, self._coordinates, 'euclidean')
        argsort_dists = np.argsort(distances, axis=1)
        # print(distances[10,:])
        distances.sort(axis=1)
        # print(distances[10,:])
        # print(argsort_dists[10,:])

        if self._eps > 0:
            ### 0-simplices
            self._simplices[0] = []
            for i in range(len(witnesses)):
                if distances[i, argsort_dists[i, 0]] <= self._eps:
                    simplex = (argsort_dists[i, 0],)
                    if simplex not in self._simplices[0]:
                        self._simplices[0].append(simplex)

            for d in range(1, max_dim + 1):
                self._simplices[d] = []
                self._barren_witnesses[d] = []
                self._weakly_witnessed[d] = []
                for i in range(len(witnesses)):
                    if distances[i, argsort_dists[i, d]] <= self._eps:
                        simplex = tuple(np.sort(argsort_dists[i, :d+1]))
                        if simplex not in self._simplices[d]:
                            well_witnessed = True
                            for face in combinations(simplex, d):
                                if face not in self._simplices[d-1]:
                                    well_witnessed = False
                                    break
                            if well_witnessed:
                                self._simplices[d].append(simplex)
                            else:
                                self._barren_witnesses[d].append(i)
                                self._weakly_witnessed[d].append(simplex)
        else:
            ### 0-simplices
            self._simplices[0] = []
            for i in range(len(witnesses)):
                # if distances[i, argsort_dists[i, 0]] <= self._eps:
                simplex = (argsort_dists[i, 0],)
                if simplex not in self._simplices[0]:
                    self._simplices[0].append(simplex)

            for d in range(1, max_dim + 1):
                self._simplices[d] = []
                self._barren_witnesses[d] = []
                self._weakly_witnessed[d] = []
                for i in range(len(witnesses)):
                    # if distances[i, argsort_dists[i, d]] <= self._eps:
                    simplex = tuple(np.sort(argsort_dists[i, :d+1]))
                    if simplex not in self._simplices[d]:
                        well_witnessed = True
                        for face in combinations(simplex, d):
                            if face not in self._simplices[d-1]:
                                well_witnessed = False
                                break
                        if well_witnessed:
                            self._simplices[d].append(simplex)
                            self._st.insert(list(simplex))
                        else:
                            self._barren_witnesses[d].append(i)
                            self._weakly_witnessed[d].append(simplex)

        self._st.compute_persistence(persistence_dim_max=True)
        self._betti_numbers = self._st.betti_numbers()

    @classmethod
    def construct(cls, st, simpls, coords, dim):
        obj = cls.__new__(cls)
        obj._st = st
        obj._simplices = simpls
        obj._coordinates = coords
        obj._dim = dim
        obj._betti_numbers = None
        return obj

    # @property
    # def betti_numbers(self):
    #     if self._st is None:
    #         self._st = SimplexTree()
    #         max_d = -1
    #         for dsimplices in self._simplices.values():
    #             for s in dsimplices:
    #                 self._st.insert(list(s))
    #                 if len(s) > max_d:
    #                     max_d = len(s)
    #         self._st.set_dimension(max_d)
    #         self._st.compute_persistence()
    #
    #     return self._st.betti_numbers()

class PatchedWitnessComplex(WitnessComplex):
    def __init__(self, landmarks, witnesses,  max_dim, eps=-1,
                 patching_level=1, max_patched_dimensions=2, patching_type="knitting"):
        super(PatchedWitnessComplex, self).__init__(landmarks, witnesses, max_dim, eps)

        distances = cdist(witnesses, self._coordinates, 'euclidean')
        argsort_dists = np.argsort(distances, axis=1)
        distances.sort(axis=1)

        # e.g. pd=2 is for closing 1-dim holes
        for pd in range(2, max_patched_dimensions + 1):
            # e.g. cpd=1 uses 3-simplices for closing 1-holes
            # e.g. cpd=2 uses 4-simplices for closing 1-holes
            for cpd in range(1, patching_level + 1):
                ### Number of vertices of the patching simplex
                nvp = pd + cpd + 1
                extra_simplices = []

                if pd not in self._barren_witnesses:
                   bwitnesses = range(len(witnesses))
                else:
                   bwitnesses = self._barren_witnesses[pd]
                for x in bwitnesses:
                    new_simplex = tuple(np.sort(argsort_dists[x, :nvp]))
                    if new_simplex in extra_simplices:
                        continue

                    hole_subcomplex = self.subcomplex(new_simplex)
                    ### Check if new_simplex is a simple cycle
                    if len(hole_subcomplex._simplices[pd]) == 0:
                        num_of_cofaces = {i: 0 for i in hole_subcomplex._simplices[pd-2]}
                        for s in hole_subcomplex._simplices[pd-1]:
                            for face in combinations(s, pd-1):
                                num_of_cofaces[face] += 1
                        if all(np.array(list(num_of_cofaces.values()))==2):
                            # print(hole_subcomplex.betti_numbers)
                            self.knitting_patching(hole_subcomplex)
                            self.merge_complex(hole_subcomplex)


    def knitting_patching(self, hole_subcomplex):
        vertices = [v for (v,) in hole_subcomplex.simplices[0]]
        vertices.sort()
        if len(vertices) == 6:
            print(6)
        potential_edges = [tuple(e) for e in combinations(vertices, 2)]
        for e in hole_subcomplex.simplices[1]:
            potential_edges.remove(e)

        # potential_edges = [tuple(e) for e in combinations(vertices, 2) if tuple(e) not in hole_subcomplex.simplices[1]]
        edges_lengths = [np.linalg.norm(self.coordinates[e1]-self.coordinates[e2]) for (e1, e2) in potential_edges]
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
