import matplotlib.pyplot as plt
import numpy as np

from CGAL.CGAL_Kernel import Point_2, Triangle_2, Segment_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from dyrect import Delaunay2d_complex, draw_complex

def p2tuple(point):
    return (point.x(), point.y())

def test_delaunay_2d():
    coords = np.array([
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1.1, 0.9],
        [-1., 0.],
        [0., -1.],
        [0.8, .8],
        [2.0, 2.],
        [0.25, 0.25]
    ])

    points = [Point_2(*cc) for cc in coords]
    points2id = {(points[i].x(), points[i].y()): i for i in range(len(points))}
    dt = Delaunay_triangulation_2()
    dt.insert(points)
    seg = Segment_2(points[0], points[1])
    tr = Triangle_2(points[0], points[1], points[2])
    print(tr.vertex(0))

    fig = plt.figure()
    ax = fig.add_subplot()

    for tr in dt.finite_faces():
        face = dt.triangle(tr)

        idx_face = [points2id[p2tuple(face.vertex(i))] for i in [0, 1, 2]]
        poly = plt.Polygon(coords[idx_face, :], alpha=0.5, color='b')
        plt.gca().add_patch(poly)

    for edge in dt.finite_edges():
        segment = dt.segment(edge)
        idx_segment = points2id[p2tuple(segment.source())], points2id[p2tuple(segment.target())]
        # print(segment.sou)
        print(idx_segment)
        ax.plot(coords[idx_segment,0], coords[idx_segment,1], linewidth=2, c='b')
    ax.scatter(coords[:, 0], coords[:, 1], c='k', s=20)
    plt.show()

def test_delaunay_complex():
    coords = np.array([
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1.1, 0.9],
        [-1., 0.],
        [0., -1.],
        [0.8, .8],
        [2.0, 2.],
        [0.25, 0.25]
    ])
    dc2 = Delaunay2d_complex(coords)
    print(dc2.betti_numbers)
    draw_complex(dc2)
    plt.show()

test_delaunay_complex()
