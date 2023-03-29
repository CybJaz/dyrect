from CGAL.CGAL_Kernel import Point_2, Point_3, Triangle_2, Triangle_3, Segment_2, Segment_3, do_intersect
from CGAL import CGAL_Convex_hull_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

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

