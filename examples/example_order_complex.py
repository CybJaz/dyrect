from dyrect import Poset, EpsilonNet, Complex, NerveComplex, draw_complex, unit_circle_sample, draw_poset
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations


def order_complex_from_poset():
    dg = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    coords = np.array([[0., 0.], [0., 1.], [1, 0.], [1., 1.]])
    poset = Poset.from_dag(dg)
    draw_poset(poset)

    order_simplices = poset.order_complex()
    print(order_simplices)
    ocomplex = Complex.construct(order_simplices, coords)
    fig, ax = draw_complex(ocomplex, circles=False)
    plt.show()

def order_complex_from_complex():
    coords = np.array([[0.,0.], [0.,1.], [-1,2.], [1.,2.]])
    simplices = {0: [(0,), (1,), (2,), (3,)],
                 1:[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
                 2:[(0,1,2), (0,1,3)]}
    complex = Complex.construct(simplices, coords)
    fig, ax = draw_complex(complex, circles=False)
    # plt.show()

    nsimplices = sum([len(x) for x in simplices.values()])
    baricenters = []
    poset = Poset(nsimplices)
    sim2idx = dict()
    simplex2baricenter = dict()
    for i, s in zip(range(nsimplices), chain(*simplices.values())):
        sim2idx[s] = i
        bc = np.average(coords[list(s),:], axis=0)
        baricenters.append(bc)
        # for d in range(1,len(s)):
        if len(s) > 1:
            for face in combinations(s, len(s)-1):
                poset.add_relation(sim2idx[face], i)
    baricentric_simplices = poset.order_complex()
    print(baricentric_simplices)
    bcomplex = Complex.construct(baricentric_simplices, np.array(baricenters))
    fig, ax = draw_complex(bcomplex, circles=False)
    plt.show()

def order_complex_from_epsilon_net():
    np.random.seed(0)
    points = unit_circle_sample(10000, noise=0.4)

    eps = 0.25
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN._landmarks
    nc = NerveComplex(lms, eps, 2, points)
    print(lms.shape)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(points[:, 0], points[:, 1], s=0.5)
    plt.scatter(lms[:, 0], lms[:, 1], s=21.9)
    fig, ax = draw_complex(nc, circles=True, ax=ax, fig=fig)

    bnc = nc.baricentric_subdivision()
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(points[:, 0], points[:, 1], s=0.5)
    plt.scatter(lms[:, 0], lms[:, 1], s=21.9)
    fig, ax = draw_complex(bnc, circles=False, ax=ax, fig=fig)
    plt.show()


if __name__ == '__main__':
    order_complex_from_poset()
    order_complex_from_complex()
    order_complex_from_epsilon_net()