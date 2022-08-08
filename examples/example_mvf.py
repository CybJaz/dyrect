import dyrect as dy
from dyrect import Complex, Poset, draw_complex, MVF, draw_poset, draw_planar_mvf, draw_transition_graph, draw_3D_mvf,\
    sampled_2d_system, double_limit_cycle, limit_cycle
import numpy as np
import matplotlib.pyplot as plt


def mvf_from_cell_complex_1_2D():
    # nerve complex
    simplices = {0: [(0,), (1,), (2,), (3,), (4,), (5,)],
                 1: [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (4, 5)],
                 2: [(0, 1, 2)]}
    coords = np.array([[0., 1.], [1., 0.], [0, 0.], [-1., 0.], [0., -1.], [0., -2.]])
    tg = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    tg = np.transpose(tg)

    # example 2
    simplices = {0: [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)],
                 1: [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 7), (4, 5), (5, 6), (6, 7), (6, 8), (7, 8)],
                 2: [(0, 1, 2), (6, 7, 8)]}
    coords = np.array([[0., 1.], [1., 0.], [0, 0.], [-1., 0.], [0., -1.], [0., -2.], [-1, -2], [-1., -1], [-2, -2]])
    tg = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                   [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1
                   [1, 0, 0, 1, 0, 0, 0, 0, 0],  # 2
                   [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 3
                   [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
                   [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6
                   [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7
                   [0, 0, 0, 0, 0, 0, 1, 0, 1]]) # 8

    # tg = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                [0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 1, 1, 1]])
    # tg = np.transpose(tg)

    nc = Complex.construct(simplices, coords)
    nerve_poset = nc.to_poset()
    # a transition graph

    # tg = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0]])

    mvf = MVF.from_cell_complex(tg, nc)
    # print(mvf)
    draw_planar_mvf(mvf)
    draw_complex(nc)
    draw_transition_graph(tg, coords, threshold=0.01, node_size=20, edge_size=18)
    plt.show()


def mvf_from_cell_complex_2D():
    # nerve complex
    simplices = {0: [(0,), (1,), (2,), (3,)],
                 1: [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
                 2: [(0, 1, 2), (1, 2, 3)]}
    coords = np.array([[0., 0.], [0., 1.], [1, 0.], [1., 1.]])
    nc = Complex.construct(simplices, coords)
    nerve_poset = nc.to_poset()
    # a transition graph
    tg = np.array([[0,1,1,0],[0,0,0,1],[0,1,0,1],[0,0,0,0]])

    # tg = np.array([[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    # tg = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0]])

    mvf = MVF.from_cell_complex(tg, nc, assigning_style='frivolous')
    # print(mvf)
    draw_planar_mvf(mvf)
    # draw_complex(nc)
    draw_transition_graph(tg, coords, threshold=0.01, node_size=20, edge_size=18)

    print("mvf: ", mvf.partition)
    for idx, v in enumerate(mvf.partition):
        print(v)
        print(mvf._conley_index[idx])
        print([mvf._point2simplex[s] for s in v])
    plt.show()


def mvf_on_tetrahedron():
    # nerve complex
    simplices = {0: [(0,), (1,), (2,), (3,)],
                 1: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                 2: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
                 3: [(0, 1, 2, 3)]}
    coords = np.array([[0., 0., 0.], [1, 0., 0.], [0., 2., 0.], [0.25, 0.5, 3.]])
    nc = Complex.construct(simplices, coords)
    nerve_poset = nc.to_poset()
    # a transition graph
    tg = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
    # tg = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
    # tg = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

    tg = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    tg = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    # tg = np.transpose(tg)
    # tg = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
    # tg = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0]])

    mvf = MVF.from_cell_complex(tg, nc, assigning_style='prudent')
    # print(mvf)
    # nc._dim = 2
    # draw_complex(nc)
    draw_3D_mvf(mvf, mode='crit')
    # draw_complex(nc)
    draw_transition_graph(tg, coords, threshold=0.01, node_size=20, edge_size=18)

    # print("mvf: ", mvf.partition)
    for idx, v in enumerate(mvf.partition):
    #     print(v)
        print(mvf._conley_index[idx])
        print([mvf._point2simplex[s] for s in v])
    for i in range(mvf.complex.nvertices):
        print(i, ": ", mvf.complex.coordinates[i])
    plt.show()


def mvf_from_cell_complex_1D():
    # nerve complex
    simplices = {0: [(0,), (1,), (2,)],
                 1: [(0, 1), (0, 2), (1, 2)]}
    coords = np.array([[0., 0.], [0., 1.], [1, 0.]])
    nc = Complex.construct(simplices, coords)
    nerve_poset = nc.to_poset()
    # a transition graph
    tg = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])

    # mvf = MVF.from_cell_complex(tg, nc)

    mvf = MVF.from_cell_complex(tg, nc)
    # draw_poset(nerve_poset)
    # draw_complex(mvf.complex)
    # draw_poset(mvf.fspace)

    draw_planar_mvf(mvf)
    # draw_transition_graph(tg, coords, threshold=0.01, node_size=20, edge_size=18)


    # print(mvf._simplex2mv)
    # fig, ax = draw_complex(nc, circles=False)
    # draw_poset(nerve_poset.get_reversed())
    plt.show()


def mvf_from_time_series():
    np.random.seed(42)
    points = np.array(
        [np.cos(t * np.pi / 6.) * np.sin(t * np.pi / 12.) * 50. for t in np.arange(0, 5000, np.sqrt(2) / 20.)])

    points = points + np.random.random(points.shape) * 2.2

    series = points
    series_emb = np.array(dy.embedding(series, 2, 20))
    # take every k-th step
    series_emb = np.take(series_emb, np.arange(0, len(series_emb), 9), axis=0)

    eps = 11.
    np.random.seed(42)
    EN = dy.EpsilonNet(eps, 0)
    EN.fit(series_emb)
    lms = EN.landmarks

    nc = dy.NerveComplex(lms, eps, 2, points=series_emb)

    TM = dy.TransitionMatrix(lms, eps, alpha=True)
    transitions = TM.transform(series_emb)
    prob_matrix = dy.trans2prob(transitions)

    GTM = dy.GeomTransitionMatrix(lms, nc, eps, alpha=True)
    transitions = GTM.fit(series_emb)
    gtm_prob_matrix = dy.trans2prob(transitions)
    gtm_prob_matrix = np.where(gtm_prob_matrix > 0.2, gtm_prob_matrix, 0.0)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    for lm in lms:
        crc = plt.Circle(lm, eps, color='r', alpha=0.05)
        ax.add_patch(crc)
    dy.draw_complex(nc, fig=fig, ax=ax)

    fig = plt.figure(figsize=(10, 8))
    rows = 2
    cols = 2

    # ax = plt.subplot(rows, cols, 1)
    # plt.scatter(series_emb[:, 0], series_emb[:, 1], s=0.2)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 1)
    plt.scatter(series_emb[:, 0], series_emb[:, 1], s=0.2)
    plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    for lm in lms:
        crc = plt.Circle(lm, eps, color='r', alpha=0.05)
        ax.add_patch(crc)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 2)
    dy.draw_complex(nc, fig=fig, ax=ax)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 3)
    dy.draw_transition_graph(prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 4)
    dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)

    # dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8)

    mvf = MVF.from_cell_complex(gtm_prob_matrix, nc, assigning_style='frivolous')
    draw_planar_mvf(mvf, mode='all')

    plt.show()


def mvf_for_lorenz():
    n = 11000
    points = dy.lorenz_attractor(n, step=0.02)

    eps = 7.
    np.random.seed(42)
    EN = dy.EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks

    nc = dy.NerveComplex(lms, eps, 3, points=points)

    TM = dy.TransitionMatrix(lms, eps, alpha=True)
    transitions = TM.transform(points)
    prob_matrix = dy.trans2prob(transitions)

    GTM = dy.GeomTransitionMatrix(lms, nc, eps, alpha=True)
    transitions = GTM.fit(points)
    gtm_prob_matrix = dy.trans2prob(transitions)
    gtm_prob_matrix = np.where(gtm_prob_matrix > 0.01, gtm_prob_matrix, 0.0)

    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
    dy.draw_complex(nc, fig=fig, ax=ax)

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(projection='3d')
    # plt.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
    # plt.scatter(lms[:, 0], lms[:, 1], lms[:, 2], s=5.2)
    # for lm in lms:
    #     crc = plt.Circle(lm, eps, color='r', alpha=0.05)
    #     ax.add_patch(crc)

    # ax = plt.subplot(rows, cols, 2)
    # dy.draw_complex(nc, fig=fig, ax=ax)
    # ax = plt.subplot(rows, cols, 3)
    # dy.draw_transition_graph(prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8)

    # ax = plt.subplot(rows, cols, 4)
    # dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)

    dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8)

    mvf = MVF.from_cell_complex(gtm_prob_matrix, nc, assigning_style='frivolous')
    draw_3D_mvf(mvf, mode='crit')

    plt.show()


def mvf_on_torus():
    points = dy.torus_rotation(100000, rotation=np.sqrt(2))
    print(points.shape)
    eps = .25

    np.random.seed(42)
    EN = dy.EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks

    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.01)
    # dy.draw_complex(nc, fig=fig, ax=ax)

    nc = dy.AlphaNerveComplex(lms, eps, 2, points=points)
    print([len(s) for s in nc.simplices.values()])
    # print(nc.betti_numbers)
    print(nc.coordinates[0])
    nc._dim = 3
    print(nc.dimension)
    dy.draw_complex(nc)
    plt.show()


def mvf_from_time_series():
    np.random.seed(42)
    points = np.array(
        [np.cos(t * np.pi / 6.) * np.sin(t * np.pi / 12.) * 50. for t in np.arange(0, 5000, np.sqrt(2) / 20.)])

    points = points + np.random.random(points.shape) * 2.2

    series = points
    series_emb = np.array(dy.embedding(series, 2, 20))
    # take every k-th step
    series_emb = np.take(series_emb, np.arange(0, len(series_emb), 9), axis=0)

    eps = 11.
    np.random.seed(42)
    EN = dy.EpsilonNet(eps, 0)
    EN.fit(series_emb)
    lms = EN.landmarks

    nc = dy.NerveComplex(lms, eps, 2, points=series_emb)

    TM = dy.TransitionMatrix(lms, eps, alpha=True)
    transitions = TM.transform(series_emb)
    prob_matrix = dy.trans2prob(transitions)

    GTM = dy.GeomTransitionMatrix(lms, nc, eps, alpha=True)
    transitions = GTM.fit(series_emb)
    gtm_prob_matrix = dy.trans2prob(transitions)
    gtm_prob_matrix = np.where(gtm_prob_matrix > 0.2, gtm_prob_matrix, 0.0)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    for lm in lms:
        crc = plt.Circle(lm, eps, color='r', alpha=0.05)
        ax.add_patch(crc)
    dy.draw_complex(nc, fig=fig, ax=ax)

    fig = plt.figure(figsize=(10, 8))
    rows = 2
    cols = 2

    # ax = plt.subplot(rows, cols, 1)
    # plt.scatter(series_emb[:, 0], series_emb[:, 1], s=0.2)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 1)
    plt.scatter(series_emb[:, 0], series_emb[:, 1], s=0.2)
    plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    for lm in lms:
        crc = plt.Circle(lm, eps, color='r', alpha=0.05)
        ax.add_patch(crc)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 2)
    dy.draw_complex(nc, fig=fig, ax=ax)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 3)
    dy.draw_transition_graph(prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)

    # plt.show()
    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot(rows, cols, 1)
    ax = plt.subplot(rows, cols, 4)
    dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)

    # dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8)

    mvf = MVF.from_cell_complex(gtm_prob_matrix, nc, assigning_style='frivolous')
    draw_planar_mvf(mvf, mode='all')

    plt.show()


def mvf_from_sampled_system():
    np.random.seed(0)
    nsp = 500
    ts = 4
    step = 0.2

    limx = 1.7
    # limx = 2.
    bounds = np.array([[-limx, limx], [-limx, limx]])

    np.random.seed(0)
    # trajectories, starting_points = sampled_2d_system(double_limit_cycle, nsp, ts, step=step, bounds=bounds)
    trajectories, starting_points = sampled_2d_system(limit_cycle, nsp, ts, step=step, bounds=bounds)
    # trajectories = sampled_2d_system(limit_cycle, nsp, ts, step=step)
    all_points = np.array([p for points in trajectories for p in points])


    eps = 0.2
    np.random.seed(2)
    EN = dy.EpsilonNet(eps, 0)
    EN.fit(all_points)
    lms = EN.landmarks
    nc = dy.AlphaNerveComplex(lms, eps, 2, points=all_points)
    # nc = dy.NerveComplex(lms, eps, 2, points=all_points)

    fig = plt.figure()
    ax = plt.subplot()
    # for tr in trajectories[:1000]:
    for tr in trajectories:
            ax.plot(tr[:, 0], tr[:, 1])

    # fig = plt.figure()
    # ax = plt.subplot()
    # dy.draw_complex(nc, fig=fig, ax=ax)
    #
    plt.show()
    fig.savefig('mvf_flow_5.png')

    # TM = dy.TransitionMatrix(lms, eps, alpha=True)
    # transitions = TM.transform(series_emb)
    # prob_matrix = dy.trans2prob(transitions)
    #
    GTM = dy.GeomTransitionMatrix(lms, nc, eps, alpha=True)
    transitions = GTM.fit(trajectories)
    gtm_prob_matrix = dy.trans2prob(transitions)
    # gtm_prob_matrix = np.where(gtm_prob_matrix > 0.02, gtm_prob_matrix, 0.0)

    # fig = plt.figure(figsize=(10, 8))
    # ax = plt.subplot()
    # plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    # for lm in lms:
    #     crc = plt.Circle(lm, eps, color='r', alpha=0.05)
    #     ax.add_patch(crc)
    # dy.draw_complex(nc, fig=fig, ax=ax)
    #
    # fig = plt.figure(figsize=(10, 8))
    # rows = 2
    # cols = 2
    #
    # # ax = plt.subplot(rows, cols, 1)
    # # plt.scatter(series_emb[:, 0], series_emb[:, 1], s=0.2)
    #
    # # plt.show()
    # # fig = plt.figure(figsize=(10, 8))
    # # ax = plt.subplot(rows, cols, 1)
    # ax = plt.subplot(rows, cols, 1)
    # plt.scatter(series_emb[:, 0], series_emb[:, 1], s=0.2)
    # plt.scatter(lms[:, 0], lms[:, 1], s=5.2)
    # for lm in lms:
    #     crc = plt.Circle(lm, eps, color='r', alpha=0.05)
    #     ax.add_patch(crc)
    #
    # # plt.show()
    # # fig = plt.figure(figsize=(10, 8))
    # # ax = plt.subplot(rows, cols, 1)
    # ax = plt.subplot(rows, cols, 2)
    # dy.draw_complex(nc, fig=fig, ax=ax)
    #
    # # plt.show()
    # # fig = plt.figure(figsize=(10, 8))
    # # ax = plt.subplot(rows, cols, 1)
    # ax = plt.subplot(rows, cols, 3)
    # dy.draw_transition_graph(prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)
    #
    # plt.show()

    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot()
    # ax = plt.subplot(rows, cols, 4)
    dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8, fig=fig, ax=ax)
    plt.show()
    fig.savefig('mvf_graph_5.png')

    # # dy.draw_transition_graph(gtm_prob_matrix, lms, threshold=0.01, node_size=10, edge_size=8)
    #
    style = 'balanced'
    # style = 'frivolous'
    # style = 'prudent'
    mvf = MVF.from_cell_complex(gtm_prob_matrix, nc, assigning_style=style)
    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot()
    draw_planar_mvf(mvf, mode='all', figsize=(15,15), fig=fig, ax=ax)
    fig.savefig('mvf_all_5.pdf')
    plt.show()
    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot()
    draw_planar_mvf(mvf, mode='crit', figsize=(15,15), fig=fig, ax=ax)
    plt.show()
    fig.savefig('mvf_crit_5.pdf')
    #

if __name__ == '__main__':
    # mvf_from_cell_complex_1D()
    # mvf_from_cell_complex_2D()
    # mvf_from_cell_complex_1_2D()
    # mvf_from_time_series()
    mvf_from_sampled_system()
    # mvf_on_tetrahedron()
    # mvf_for_lorenz()
    # mvf_on_torus()
