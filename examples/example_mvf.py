from dyrect import Complex, Poset, draw_complex, MVF, draw_poset, draw_planar_mvf
import numpy as np
import matplotlib.pyplot as plt

def mvf_from_cell_complex_2D():
    # nerve complex
    simplices = {0: [(0,), (1,), (2,), (3,)],
                 1: [(0,1), (0,2), (1,2), (1,3), (2,3)],
                 2: [(0,1,2), (1,2,3)]}
    coords = np.array([[0., 0.], [0., 1.], [1, 0.], [1., 1.]])
    nc = Complex.construct(simplices, coords)
    nerve_poset = nc.to_poset()
    # a transition graph
    # tg = np.array([[0,1,1,0],[0,0,0,1],[0,1,0,1],[0,0,0,0]])

    tg = np.array([[0,1,0,0],[0,0,1,1],[1,0,0,0],[0,0,1,0]])
    # tg = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0]])

    mvf = MVF.from_cell_complex(tg, nc)
    print(mvf)
    draw_planar_mvf(mvf)
    plt.show()

def mvf_from_cell_complex_1D():
    # nerve complex
    simplices = {0: [(0,), (1,), (2,)],
                 1: [(0,1), (0,2), (1,2)]}
    coords = np.array([[0., 0.], [0., 1.], [1, 0.]])
    nc = Complex.construct(simplices, coords)
    nerve_poset = nc.to_poset()
    # a transition graph
    tg = np.array([[0,1,1],[0,0,0],[0,1,0]])

    # mvf = MVF.from_cell_complex(tg, nc)

    mvf = MVF.from_cell_complex(tg, nc)
    # draw_poset(nerve_poset)
    # draw_complex(mvf.complex)
    # draw_poset(mvf.fspace)

    draw_planar_mvf(mvf)

    print(mvf._simplex2mv)
    # fig, ax = draw_complex(nc, circles=False)
    # draw_poset(nerve_poset.get_reversed())
    plt.show()

if __name__ == '__main__':
    # mvf_from_cell_complex_1D()
    # mvf_from_cell_complex_1D()
    mvf_from_cell_complex_2D()