from CGAL.CGAL_Kernel import Point_2, Point_3, Triangle_2, Triangle_3, Segment_2, Segment_3, do_intersect
from CGAL import CGAL_Convex_hull_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup


from itertools import combinations, permutations
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from .drawing import draw_complex

from .complex import Complex
from gudhi import SimplexTree

# def cgal_2D_convexity_test():
#     a = Point_2(0., 0.)
#     b = Point_2(1., 0.)
#     c = Point_2(0., 1.)
#     d = Point_2(1.1, 0.9)
#
#     out = []
#     CGAL_Convex_hull_2.convex_hull_2([a,b,c,d], out)
#     for p in out:
#         print(p)

# def cgal_2D_intersections_test():
#     a = Point_3(0., 0., 0.)
#     b = Point_3(1., 0., 0.)
#     c = Point_3(0., 1., 0.)
#     d = Point_3(1.1, 0.9, 0.)
#     e = Point_3(-1., 0., 0.)
#     f = Point_3(0., -1., 0.)
#     g = Point_3(0.8, .8, 0.)
#     h = Point_3(2.0, 2., 0.)
#     i = Point_3(0.25, 0.25, 0.)
#
#     ad = Segment_3(a, d)
#     gh = Segment_3(h, g)
#     fi = Segment_3(i, f)
#     ef = Segment_3(e, f)
#     ab = Line_3(a, b)
#     bc = Line_3(b, c)
#     cd = Line_3(c, d)
#
#     aef = Triangle_3(a, e, f)
#     abc = Triangle_3(a, b, c)
#     abd = Triangle_3(a, b, d)
#     bcd = Triangle_3(b, c, d)
#
#     # constructs AABB tree
#     tree = AABB_tree_Triangle_3_soup([abc, abd, bcd, aef])
#
#     # print(tree.any_intersected_primitive(abc))
#     out = []
#     print(tree.all_intersections(gh, out))
#     print([t[1] for t in out])
#     print(abc.has_on(i))
#     print(abc.has_on(h))
#     print(do_intersect(abc, ab))
#     print(do_intersect(abc, fi))
#     print(do_intersect(abc, fi))
#     print(do_intersect(abc, ef))
#     # print([Triangle_3(t) for t in out])
#     # print(tree.number_of_intersected_primitives(abc),
#     #       " intersections(s) with ray query")
#

def p2tuple(point):
    return (point.x(), point.y())
def p3tuple(point):
    return (point.x(), point.y(), point.z())

class Delaunay2d_complex(Complex):
    def __init__(self, landmarks):
        self._st= SimplexTree()
        self._simplices = dict()
        self._ambient_dim = 2
        self._coordinates = {idx: coord[:2] for idx, coord in enumerate(landmarks)}

        p2id = {(x, y): i for i, [x, y] in enumerate(landmarks)}
        dt = Delaunay_triangulation_2()
        cgal_points = [Point_2(*cc) for cc in landmarks]
        dt.insert(cgal_points)

        self._simplices[0] = [(v,) for v in range(len(landmarks))]
        self._simplices[1] = []
        self._simplices[2] = []

        for edge in dt.finite_edges():
            segment = dt.segment(edge)
            idx_edge = [p2id[p2tuple(segment.source())], p2id[p2tuple(segment.target())]]
            idx_edge.sort()
            self._simplices[1].append(tuple(idx_edge))
        for tr in dt.finite_faces():
            face = dt.triangle(tr)
            idx_face = [p2id[p2tuple(face.vertex(i))] for i in [0, 1, 2]]
            idx_face.sort()
            self._simplices[2].append(tuple(idx_face))

        for d in self._simplices:
            for s in self._simplices[d]:
                self._st.insert(s)
        self._st.compute_persistence(persistence_dim_max=True)
        self._betti_numbers = self._st.betti_numbers()

class Delaunay3d_complex(Complex):
    def __init__(self, landmarks):
        self._st= SimplexTree()
        self._simplices = dict()
        self._ambient_dim = 3
        self._coordinates = {idx: coord[:3] for idx, coord in enumerate(landmarks)}

        p2id = {(x, y, z): i for i, [x, y, z] in enumerate(landmarks)}
        dt = Delaunay_triangulation_3()
        cgal_points = [Point_3(*cc) for cc in landmarks]
        dt.insert(cgal_points)

        self._simplices[0] = [(v,) for v in range(len(landmarks))]
        self._simplices[1] = []
        self._simplices[2] = []

        for edge in dt.finite_edges():
            segment = dt.segment(edge)
            idx_edge = [p2id[p3tuple(segment.source())], p2id[p3tuple(segment.target())]]
            idx_edge.sort()
            self._simplices[1].append(tuple(idx_edge))
        for tr in dt.finite_facets():
            face = dt.triangle(tr)
            idx_face = [p2id[p3tuple(face.vertex(i))] for i in [0, 1, 2]]
            idx_face.sort()
            self._simplices[2].append(tuple(idx_face))

        for d in self._simplices:
            for s in self._simplices[d]:
                self._st.insert(s)
        self._st.compute_persistence(persistence_dim_max=True)
        self._betti_numbers = self._st.betti_numbers()

def is_convex_hull(points):
    if len(points[0]) == 2:
        cgal_points = [Point_2(*p) for p in points]
        convex_hull = []
        CGAL_Convex_hull_2.convex_hull_2(cgal_points, convex_hull)
        is_convex = True
        for p in cgal_points:
            # print(p, p in convex_hull)
            if p not in convex_hull:
                is_convex = False
        return is_convex


def in_convex_hull(points):
    if len(points[0]) == 2:
        cgal_points = [Point_2(*p) for p in points]
        convex_hull = []
        CGAL_Convex_hull_2.convex_hull_2(cgal_points, convex_hull)
        is_convex = True
        for ip, p in enumerate(cgal_points):
            if p not in convex_hull:
                return ip
        return None

def all_triangles_intersection_test_2D(complex):
    intersecting_pairs = []
    # intersection test
    points = []
    triangles = []
    for v in complex.coordinates.values():
        points.append(Point_2(v[0], v[1]))
    for tr in complex.simplices[2]:
        (v0, v1, v2) = tr
        triangles.append(Triangle_2(points[v0], points[v1], points[v2]))
    n_triangles = len(triangles)
    for itr1, tr1 in enumerate(complex.simplices[2]):
        for itr2 in range(itr1+1, n_triangles):
            if do_intersect(triangles[itr1], triangles[itr2]):
                tr2 = complex.simplices[2][itr2]
                common_vertices = set(tr1).intersection(tr2)

                intersected_q = True
                # if len(common_vertices) == 3:
                #     intersected_q = False
                if len(common_vertices) == 2:
                    v_tr1 = [v for v in tr1 if v not in common_vertices][0]
                    v_tr2 = [v for v in tr2 if v not in common_vertices][0]
                    intersected_q = do_intersect(triangles[itr1], points[v_tr2]) or \
                                    do_intersect(triangles[itr2], points[v_tr1])
                    # intersected_q = triangles[itr1].has_on(points[v_tr2]) or \
                    #                 triangles[itr2].has_on(points[v_tr1])
                elif len(common_vertices) == 1:
                    if tr1 == (4, 9, 12):
                        print("hello")
                    seg_tr1 = Segment_2(*[points[v] for v in tr1 if v not in common_vertices])
                    seg_tr2 = Segment_2(*[points[v] for v in tr2 if v not in common_vertices])
                    intersected_q = do_intersect(triangles[itr1], seg_tr2) or \
                                    do_intersect(triangles[itr2], seg_tr1)
                    # print(intersected_q)
                if intersected_q:
                    intersecting_pairs.append((tr1, tr2))
                    print(tr1, tr2)
    print("Number of intersecting pairs: " + str(len(intersecting_pairs)))
    return intersecting_pairs

def all_triangles_intersection_test_3D(complex):
    intersecting_pairs = []
    # intersection test
    points = []
    triangles = []
    for v in complex.coordinates.values():
        # print(v)
        points.append(Point_3(v[0], v[1], v[2]))
    for tr in complex.simplices[2]:
        (v0, v1, v2) = tr
        triangles.append(Triangle_3(points[v0], points[v1], points[v2]))
    n_triangles = len(triangles)
    for itr1, tr1 in enumerate(complex.simplices[2]):
        for itr2 in range(itr1+1, n_triangles):
            if do_intersect(triangles[itr1], triangles[itr2]):
                tr2 = complex.simplices[2][itr2]
                common_vertices = set(tr1).intersection(tr2)

                intersected_q = True
                if len(common_vertices) == 3:
                    intersected_q = False
                elif len(common_vertices) == 2:
                    v_tr1 = [v for v in tr1 if v not in common_vertices][0]
                    v_tr2 = [v for v in tr2 if v not in common_vertices][0]
                    intersected_q = do_intersect(triangles[itr1], points[v_tr2]) or \
                                    do_intersect(triangles[itr2], points[v_tr1])
                    # intersected_q = triangles[itr1].has_on(points[v_tr2]) or \
                    #                 triangles[itr2].has_on(points[v_tr1])
                elif len(common_vertices) == 1:
                    seg_tr1 = Segment_3(*[points[v] for v in tr1 if v not in common_vertices])
                    seg_tr2 = Segment_3(*[points[v] for v in tr2 if v not in common_vertices])
                    intersected_q = do_intersect(triangles[itr1], seg_tr2) or \
                                    do_intersect(triangles[itr2], seg_tr1)
                    # print(intersected_q)
                if intersected_q:
                    intersecting_pairs.append((tr1, tr2))
                    print(tr1, tr2)
    print("Number of intersecting pairs: " + str(len(intersecting_pairs)))
    return intersecting_pairs


def draw_voronoi_cells_2d(lms, order=1, areax=[0, 1], areay=[0,1], resolution=10,
                          complex=None, labels=True, fig=None, ax=None):
    fwidth = 10
    # fig = plt.figure(figsize=(fwidth, fwidth*0.4))
    # rows = 1
    # cols = 2

    xpoints = np.linspace(areax[0], areax[1], resolution+1)
    ypoints = np.linspace(areay[0], areay[1], resolution+1)
    X, Y = np.meshgrid(xpoints, ypoints)

    xy = np.vstack((X.flatten(), Y.flatten())).T
    grid_dists = cdist(xy, lms, 'euclidean')
    # points_dists = cdist(points, lms, 'euclidean')
    def get_points_of_interest(verts, dim):
        sim_wareas = dict()
        sim_witnesses = dict()
        for sim in combinations(verts, dim):
            sim_wareas[tuple(np.sort(sim))] = []
            sim_witnesses[tuple(np.sort(sim))] = []
        for x_arg_sorted, x in zip(np.argsort(grid_dists, axis=1), range(len(xy))):
            wit_sim = tuple(np.sort(x_arg_sorted[:dim]))
            if wit_sim in sim_wareas:
                sim_wareas[wit_sim].append(x)
        # for x_arg_sorted, x in zip(np.argsort(points_dists, axis=1), range(len(points))):
        #     wit_sim = tuple(np.sort(x_arg_sorted[:dim + 1]))
        #     if wit_sim in sim_witnesses:
        #         sim_witnesses[wit_sim].append(x)
        return sim_wareas, sim_witnesses

    if (fig is None) or (ax is None):
        fig = plt.figure(figsize=(fwidth, fwidth))
        ax = plt.subplot()
    sim_wareas, sim_witnesses = get_points_of_interest(range(len(lms)), order)

    empty_keys = [key for key in sim_wareas.keys() if len(sim_wareas[key]) == 0]
    for key in empty_keys:
        sim_wareas.pop(key)
    if complex is not None:
        not_in_complex = [key for key in sim_wareas.keys() if key not in complex.simplices[order-1]]
    # print(not_in_complex)

    cm = plt.cm.get_cmap('nipy_spectral')(np.linspace(0.05, 1, len(sim_wareas), endpoint=False))
    np.random.seed(0)
    np.random.shuffle(cm)
    # print([(s, len(sim_witnesses[s])) for s in sim_witnesses])
    for isim, sim in enumerate(sim_wareas):
        points_list = sim_wareas[sim]
        if len(points_list) > 0:
            e_xy = np.array([xy[i, :] for i in points_list])
            # if complex is not None and sim in not_in_complex:
            #     area_color = 'k'
            #     ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, color=area_color)
            # else:
            #     area_color = cm[isim]
            #     ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, label=sim, color=area_color)
            area_color = cm[isim]
            ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, label=sim, color=area_color)
    ax.scatter(lms[:, 0], lms[:, 1], color='k', s=10.5)
    # ax.scatter(points[:, 0], points[:, 1], color='k', s=0.5)
    ax.set_aspect('equal')
    if labels:
        for v in range(len(lms)):
            ax.annotate(str(v), (lms[v, 0], lms[v, 1]), fontsize=15, bbox=dict(boxstyle="round4", fc="w"))
            legend = ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
        for i in range(len(legend.legendHandles)):
            legend.legendHandles[i]._sizes = [30]
    plt.title(str(order) + "-Voronoi cells")
    plt.xlim(areax)
    plt.ylim(areay)
    if complex is not None:
        draw_complex(complex, fig=fig, ax=ax, alpha=0.2)
    return fig, ax

def draw_directed_voronoi_cells_2d(lms, order=2, areax=[0, 1], areay=[0,1], resolution=10,
                          complex=None, labels=True, fig=None, ax=None):
    fwidth = 10
    # fig = plt.figure(figsize=(fwidth, fwidth*0.4))
    # rows = 1
    # cols = 2

    xpoints = np.linspace(areax[0], areax[1], resolution+1)
    ypoints = np.linspace(areay[0], areay[1], resolution+1)
    X, Y = np.meshgrid(xpoints, ypoints)

    xy = np.vstack((X.flatten(), Y.flatten())).T
    grid_dists = cdist(xy, lms, 'euclidean')
    # points_dists = cdist(points, lms, 'euclidean')
    def get_points_of_interest(verts, dim):
        sim_wareas = dict()
        sim_witnesses = dict()
        for sim in permutations(verts, dim):
            sim_wareas[tuple(sim)] = []
            sim_witnesses[tuple(sim)] = []
        for x_arg_sorted, x in zip(np.argsort(grid_dists, axis=1), range(len(xy))):
            wit_sim = tuple(x_arg_sorted[:dim])
            if wit_sim in sim_wareas:
                sim_wareas[wit_sim].append(x)
            else:
                print("what?")
        # for x_arg_sorted, x in zip(np.argsort(points_dists, axis=1), range(len(points))):
        #     wit_sim = tuple(np.sort(x_arg_sorted[:dim + 1]))
        #     if wit_sim in sim_witnesses:
        #         sim_witnesses[wit_sim].append(x)
        return sim_wareas, sim_witnesses

    if (fig is None) or (ax is None):
        fig = plt.figure(figsize=(fwidth, fwidth))
        ax = plt.subplot()
    sim_wareas, sim_witnesses = get_points_of_interest(range(len(lms)), order)

    empty_keys = [key for key in sim_wareas.keys() if len(sim_wareas[key]) == 0]
    for key in empty_keys:
        sim_wareas.pop(key)
    if complex is not None:
        not_in_complex = [key for key in sim_wareas.keys() if key not in complex.simplices[order-1]]

    cm = plt.cm.get_cmap('nipy_spectral')(np.linspace(0.05, 1, len(sim_wareas), endpoint=False))
    np.random.seed(0)
    np.random.shuffle(cm)
    # print([(s, len(sim_witnesses[s])) for s in sim_witnesses])
    for isim, sim in enumerate(sim_wareas):
        points_list = sim_wareas[sim]
        if len(points_list) > 0:
            e_xy = np.array([xy[i, :] for i in points_list])
            # if complex is not None and sim in not_in_complex:
            #     area_color = 'k'
            #     ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, color=area_color)
            # else:
            #     area_color = cm[isim]
            #     ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, label=sim, color=area_color)
            area_color = cm[isim]
            ax.scatter(e_xy[:, 0], e_xy[:, 1], s=.05, label=sim, color=area_color)
    ax.scatter(lms[:, 0], lms[:, 1], color='k', s=10.5)
    # ax.scatter(points[:, 0], points[:, 1], color='k', s=0.5)
    ax.set_aspect('equal')
    if labels:
        for v in range(len(lms)):
            ax.annotate(str(v), (lms[v, 0], lms[v, 1]), fontsize=15, bbox=dict(boxstyle="round4", fc="w"))
            legend = ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
        for i in range(len(legend.legendHandles)):
            legend.legendHandles[i]._sizes = [30]
    plt.title("Directed " + str(order) + "-Voronoi cells")
    plt.xlim(areax)
    plt.ylim(areay)
    if complex is not None:
        draw_complex(complex, fig=fig, ax=ax, alpha=0.2)
    return fig, ax
