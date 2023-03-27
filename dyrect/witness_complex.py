import functools
from typing import List, Any

import numpy as np
from gudhi import SimplexTree
from itertools import chain, combinations
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial.distance import cdist

from .epsilon_net import Complex


class WitnessComplex(Complex):
    def __init__(self, landmarks, witnesses, max_dim, eps=-1):
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
        # self._coordinates = landmarks
        self._eps = eps
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

        # print(np.array(list(self._coordinates.values())))
        distances = cdist(witnesses, np.array(list(self._coordinates.values())), 'euclidean')
        argsort_dists = np.argsort(distances, axis=1)
        distances.sort(axis=1)

        if self._eps > 0:
            # assert False, "an option with positive epsilon needs to be tested"
            ### 0-simplices
            self._simplices[0] = []
            for i in range(len(witnesses)):
                if distances[i, 0] <= self._eps:
                    simplex = (argsort_dists[i, 0],)
                    if simplex not in self._simplices[0]:
                        self._simplices[0].append(simplex)

            for d in range(1, max_dim + 1):
                self._simplices[d] = []
                self._barren_witnesses[d] = []
                self._weakly_witnessed[d] = []
                for i in range(len(witnesses)):
                    if distances[i, d] <= self._eps:
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
                            self._weakly_witnessed[d].append(simplex)

        self._st.compute_persistence(persistence_dim_max=True)
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

class PatchedWitnessComplex(WitnessComplex):
    def __init__(self, landmarks, witnesses, max_dim, eps=-1,
                 patching_level=1, max_patched_dimensions=2, patching_type="knitting"):
        super(PatchedWitnessComplex, self).__init__(landmarks, witnesses, max_dim, eps)

        distances = cdist(witnesses, np.array(list(self._coordinates.values())), 'euclidean')
        argsort_dists = np.argsort(distances, axis=1)
        distances.sort(axis=1)

        # TODO: incorporate eps into the patching
        if eps > 0:
            print("WARNING: eps is not taken into the account during the patching yet")

        # Patched dimension
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
                        num_of_cofaces = {i: 0 for i in hole_subcomplex._simplices[pd - 2]}
                        for s in hole_subcomplex._simplices[pd - 1]:
                            for face in combinations(s, pd - 1):
                                num_of_cofaces[face] += 1
                        if all(np.array(list(num_of_cofaces.values())) == 2):
                            # print(hole_subcomplex.betti_numbers)
                            # self.knitting_patching(hole_subcomplex)
                            # self.web_patching(hole_subcomplex)
                            self.fan_patching(hole_subcomplex)
                            # self.morse_patching(hole_subcomplex)
                            self.merge_complex(hole_subcomplex)

    def fan_patching(self, hole_subcomplex):
        vertices = [v for (v,) in hole_subcomplex.simplices[0]]
        vertices.sort()
        v0 = vertices[0]

        dims = list(hole_subcomplex.simplices.keys())
        dims.sort(reverse=True)
        hole_subcomplex.simplices[dims[0]+1] = []
        for d in dims:
            for s in hole_subcomplex.simplices[d]:
                new_simplex = list(s) + [v0]
                new_simplex.sort()
                if new_simplex not in hole_subcomplex.simplices[d+1]:
                    hole_subcomplex.simplices[d+1].append(tuple(new_simplex))
        hole_subcomplex.simplices[0].append((v0,))

    def web_patching(self, hole_subcomplex):
        vertices = [v for (v,) in hole_subcomplex.simplices[0]]
        vertices.sort()

        # coords_center = np.mean([self.coordinates[v] for v in vertices], axis=0)
        coords_center = np.mean(self.coords_list(vertices), axis=0)
        idx_center = len(self.simplices[0])
        hole_subcomplex._coordinates[idx_center] = coords_center
        print(coords_center)

        dims = list(hole_subcomplex.simplices.keys())
        dims.sort(reverse=True)
        hole_subcomplex.simplices[dims[0]+1] = []
        for d in dims:
            for s in hole_subcomplex.simplices[d]:
                new_simplex = list(s) + [idx_center]
                new_simplex.sort()
                hole_subcomplex.simplices[d+1].append(tuple(new_simplex))
        hole_subcomplex.simplices[0].append((idx_center,))


    # ### UNFINISHED METHOD!
    # def morse_patching(self, hole_subcomplex):
    #     """
    #     @param hole_subcomplex:
    #     @return:
    #     """
    #     vertices = [v for (v,) in hole_subcomplex.simplices[0]]
    #     vertices.sort()
    #     edges = [tuple(e) for e in combinations(vertices, 2)]
    #     edge_lengths = {e: np.linalg.norm(self.coordinates[e[0]] - self.coordinates[e[1]]) for e in edges}
    #     edge_type = {e: 0 if e in hole_subcomplex.simplices[1] else 1 for e in edges}
    #
    #     # print(vertices)
    #
    #     def get_signature(simplex):
    #         root_part = []
    #         root_types = []
    #         patch_part = []
    #         patch_types = []
    #         for e in combinations(simplex, 2):
    #             if edge_type[e] == 1:
    #                 patch_part.append(edge_lengths[e])
    #             else:
    #                 root_part.append(edge_lengths[e])
    #         # argsort_root = np.argsort(root_part)
    #         # argsort_patch = np.argsort(patch_part)
    #         # print(list(np.take(patch_part, argsort_patch)))
    #         # print(list(np.take(root_part, argsort_root)))
    #         # print(list(np.take(patch_part, argsort_patch)) + list(np.take(root_part, argsort_root)))
    #         patch_part.sort(reverse=True)
    #         root_part.sort(reverse=True)
    #
    #         return (len(simplex) - 1,
    #                 # list(np.take(patch_part, argsort_patch)) + list(np.take(root_part, argsort_root)),
    #                 patch_part + root_part,
    #                 list(np.ones((len(patch_part),))) + list(np.zeros((len(root_part),)))
    #                 )
    #
    #     def compare_signatures(s1, s2):
    #         if s1[0] > s2[0]:
    #             return 1
    #         elif s1[0] < s2[0]:
    #             return -1
    #         else:
    #             # if len(s1[2]) != len(s2[2]):
    #             #     print('huh')
    #             for i in range(len(s1)):
    #                 if s1[2][i] > s2[2][i]:
    #                     return 1
    #                 elif s1[2][i] < s2[2][i]:
    #                     return -1
    #                 else:
    #                     if s1[1][i] > s2[1][i]:
    #                         return 1
    #                     elif s1[1][i] < s2[1][i]:
    #                         return -1
    #         return 0
    #
    #     root = hole_subcomplex.simplices[0]
    #     patch = []
    #     signature = {}
    #     for d in range(1, len(vertices)):
    #         if d in hole_subcomplex.simplices:
    #             for s in combinations(vertices, d + 1):
    #                 signature[s] = get_signature(s)
    #                 if s not in hole_subcomplex.simplices[d]:
    #                     patch.append(s)
    #                     # print(s, signature[s])
    #                 else:
    #                     root.append(s)
    #         else:
    #             for s in combinations(vertices, d + 1):
    #                 patch.append(s)
    #                 signature[s] = get_signature(s)
    #                 print(s, signature[s])
    #
    #     def compare_simplices(s1, s2):
    #         return compare_signatures(signature[s1], signature[s2])
    #     queue = sorted(list(patch), key=functools.cmp_to_key(compare_simplices), reverse=True)
    #     # print(queue)
    #
    #     flag = True
    #     # while flag:
    #     #     top_simplex = queue.pop()
    #     #     print(top_simplex)
    #     #     if

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
