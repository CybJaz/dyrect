import dyrect as dy
# from pyinform import transfer_entropy
from itertools import combinations, combinations_with_replacement
import numpy as np
import numba
import pandas as pd
import matplotlib.pyplot as plt

out_dir = "/Users/cybjaz/workspace/dyrect/experiments/tent_logi/"


def f_label(x):
    return str(np.round(x, 3))


def lorenz_test():
    n = 10000
    k = 10
    dl = 5
    np.random.seed(0)
    lorenz = dy.lorenz_attractor(n)
    print(lorenz[:, 0].reshape((n, 1)).shape)
    lx = lorenz[:, 0].reshape((n, 1))
    ly = lorenz[:, 1].reshape((n, 1))
    lz = lorenz[:, 2].reshape((n, 1))

    rlx = dy.embedding(lx, 3, dl)[:, :, 0]
    rly = dy.embedding(ly, 3, dl)[:, :, 0]
    rlz = dy.embedding(lz, 3, dl)[:, :, 0]


# homeomorphism from a logistic map to a tent map
def h_log_to_tent(x):
    return 2 * np.arcsin(np.sqrt(x)) / np.pi
    # return 2 * sympy.asin(sympy.sqrt(x)) / sympy.pi


# homeomorphism from a tent map to a logistic map
def h_tent_to_log(x):
    # return sympy.sin(sympy.pi * x / 2) ** 2
    return np.sin(np.pi * x / 2) ** 2


# identity map
def id(x):
    return x


# a projection of x onto its i-th coordinate
def proj(x, i):
    return x[:, i]


def angle_power_interval(x, p):
    return np.power(x, p)


def perturbation_on_circle_interval(x, pl):
    """
    @param x: points from interval [0,1] interpreted as a circle
    @param pl: perturbation_level
    @return:
    """
    perturbations = (np.random.random(len(x)) * 2 * pl) - pl
    return np.mod(x + perturbations, 1.)


@numba.jit(nopython=True, fastmath=True)
def sphere_max_dist(x, y):
    modx = np.mod(x - y, 1.)
    mody = np.mod(y - x, 1.)
    return min(modx[0], mody[0])


def generate_log_tent_data(n=500, starting_points=None, log_parameters=None):
    # starting point
    # starting_points = [sympy.Rational(20, 100), sympy.Rational(21, 100), sympy.Rational(25, 100)]
    # parameters = [sympy.Rational(4, 1)]
    if starting_points is None:
        starting_points = [0.2]
    if log_parameters is None:
        log_parameters = [4.]

    data = {}
    for sp in starting_points:
        for p in log_parameters:
            print((sp, p))
            logi = np.array(dy.logistic_map(n, r=p, starting_point=sp))

            if p == 4.0:
                # tent = np.array(dy.logistic_map(n, r=4.00, starting_point=sp))
                # tent = np.array([h_log_to_tent(x).evalf() for x in logi], dtype=float)
                tent = np.array([h_log_to_tent(x) for x in logi], dtype=float)
                # logi = np.array([x.evalf() for x in logi], dtype=float)
                data[("tent", p, sp)] = tent

            data[("logm", p, sp)] = logi
    return data


def experiment_log_tent():
    np.random.seed(0)
    n = 2000

    # FNN, KNN, conjTest experiment
    # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.999, 3.99, 3.8])
    # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.99, 3.9, 3.8])
    # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0, 3.999, 3.8])
    data = generate_log_tent_data(n, starting_points=[0.2, 0.21], log_parameters=[4.0, 3.99])
    # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0])
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())
    kv = [1, 3, 5]
    # kv = [10]
    tv = [1, 3, 5, 10]
    rv = [1, 2, 3]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == 'logm' and k2[0] == 'tent':
            return h_log_to_tent
        elif k1[0] == 'tent' and k2[0] == 'logm':
            return h_tent_to_log
        else:
            return id

    experiment(data, 'log_tent_n' + str(n), kv, tv, rv, False, True, homeo)


def experiment_log_rv_grid():
    npoints = 2000
    log_params = np.arange(4.,3.8, -0.01)
    total_steps = len(log_params)
    data = generate_log_tent_data(n=2000, starting_points=[0.2], log_parameters=log_params)
    kv = [1, 3, 5, 10]
    rv = [1, 2, 3]
    # tv = list(range(1, 20, 5))
    # tv = [1, 3, 5, 10, 15, 20]
    tv = [1, 3, 5, 10]
    data.pop(list(data.keys())[0])
    keys = [k for k in data.keys()]

    conj_diffs = np.zeros((len(keys), len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys), len(kv), 2))
    fnns_diffs = np.zeros((len(keys), len(rv), 2))

    k1 = keys[0]
    base_name = 'log_params_grid_' + str(int(total_steps)) + '_'
    for j in range(0, len(keys)):
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        conj_diffs[j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], id, k=kv, t=tv)
        knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
        knns_diffs[j, :, 0] = knn1
        knns_diffs[j, :, 1] = knn2
        fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv)
        fnns_diffs[j, :, 0] = fnn1
        fnns_diffs[j, :, 1] = fnn2

    for it, t in enumerate(tv):
        conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
                               columns=[f_label(r) for r in log_params])
        strt = str(t) if t > 9 else '0' + str(t)
        conj_df.to_csv(out_dir + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
        print(conj_df.to_markdown())

    knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in log_params])
    knns1_df.to_csv(out_dir + base_name + '_knn1' + '_n' + str(npoints) + '.csv')
    knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in log_params])
    knns2_df.to_csv(out_dir + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
    print(knns1_df.to_markdown())
    print(knns2_df.to_markdown())

    fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in log_params])
    fnns1_df.to_csv(out_dir + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
    fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in log_params])
    fnns2_df.to_csv(out_dir + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')


# I keep it only because of transfer entropy test here
# def _old_experiment_log_tent():
#     np.random.seed(0)
#     n = 2000
#
#     # FNN, KNN, conjTest experiment
#     # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.999, 3.99, 3.8])
#     # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.99, 3.9, 3.8])
#     # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0, 3.999, 3.8])
#     data = generate_log_tent_data(n, starting_points=[0.2, 0.21], log_parameters=[4.0, 3.99])
#     # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0])
#     keys = [k for k in data.keys()]
#     labels = [str(k) for k in keys]
#     print(data.keys())
#     kv = [1, 3, 5]
#     kv = [10]
#     tv = [1, 3, 5, 10, 15]
#     rv = [1, 2, 3]
#
#     knn_diffs = np.zeros((len(data), len(data), len(kv)))
#     conj_diffs = np.zeros((len(data), len(data), len(kv), len(tv)))
#     fnn_diffs = np.zeros((len(data), len(data), len(rv)))
#     te_diffs = np.zeros((len(data), len(data), len(kv)))
#
#     # plt.figure()
#     # x = data[keys[0]]
#     # plt.plot(x)
#     # plt.show()
#
#     # Experiment
#     # for (i, j) in combinations(range(len(data)), 2):
#     for (i, j) in combinations_with_replacement(range(len(data)), 2):
#         k1 = keys[i]
#         k2 = keys[j]
#         print(k1, k2)
#         ts1 = data[k1].reshape((n, 1))
#         ts2 = data[k2].reshape((n, 1))
#         new_n = min(len(ts1), len(ts2))
#
#         # Transfer entropy
#         for ik, k in enumerate(kv):
#             # print(ts1[:, 0].shape)
#             b = 4
#
#             def to_binary_ts(ts, b):
#                 bins = np.linspace(0, 1, b + 1)
#                 bts = np.array([np.argmax(bins > x) for x in ts])
#                 return bts
#
#             bts1 = to_binary_ts(ts1, b)
#             bts2 = to_binary_ts(ts2, b)
#             # print(bts1)
#             print(transfer_entropy(bts1[1:], bts2[:-1], k=k))
#             print(transfer_entropy(bts1, bts2, k=k))
#             vmax_ij = -1
#             vmax_ji = -1
#             argmax_ij = -1
#             argmax_ji = -1
#             for shift in range(1, 100):
#                 v_ij = transfer_entropy(bts1[shift:], bts2[:-shift], k=k)
#                 if vmax_ij < v_ij:
#                     vmax_ij = v_ij
#                     argmax_ij = shift
#                 if i != j:
#                     v_ji = transfer_entropy(bts2[shift:], bts1[:-shift], k=k)
#                     if vmax_ji < v_ji:
#                         vmax_ji = v_ji
#                         argmax_ji = shift
#                 # print(shift, bts1[shift:].shape, bts2[:-shift].shape)
#             te_diffs[i, j, ik] = vmax_ij
#             if i != j:
#                 te_diffs[j, i, ik] = vmax_ji
#             print(argmax_ij, (argmax_ji if j != i else None))
#
#         # KNN test
#         # knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
#         # knn_diffs[i, j, :] = knn1
#         # knn_diffs[j, i, :] = knn2
#         #
#         # FNN test
#         # fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv)
#         # fnn_diffs[i, j, :] = fnn1
#         # fnn_diffs[j, i, :] = fnn2
#
#         # Conj test
#         # if k1[0] == 'logm' and k2[0] == 'tent':
#         #     h1 = h_log_to_tent
#         #     h2 = h_tent_to_log
#         # elif k1[0] == 'tent' and k2[0] == 'logm':
#         #     h1 = h_tent_to_log
#         #     h2 = h_log_to_tent
#         # else:
#         #     h1 = id
#         #     h2 = id
#         #
#         # conj_diffs[i, j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], h1, k=kv, t=tv)
#         # conj_diffs[j, i, :, :] = dy.conjugacy_test(ts2[:new_n], ts1[:new_n], h2, k=kv, t=tv)
#
#     # Transfer Entropy results to files
#     for ik, k in enumerate(kv):
#         te_df = pd.DataFrame(data=te_diffs[:, :, ik], index=labels, columns=labels)
#         # knn_df.to_csv(out_dir + 'tent_logi_knns_k' + str(k) + '_n' + str(n) + '.csv')
#         print(te_df.to_markdown())
#
#     # KNN results to files
#     # for ik, k in enumerate(kv):
#     #     knn_df = pd.DataFrame(data=knn_diffs[:, :, ik], index=labels, columns=labels)
#     #     knn_df.to_csv(out_dir + 'tent_logi_knns_k' + str(k) + '_n' + str(n) + '.csv')
#     #     print(knn_df.to_markdown())
#
#     # FNN results to files
#     # for ir, r in enumerate(rv):
#     #     fnn_df = pd.DataFrame(data=fnn_diffs[:, :, ir], index=labels, columns=labels)
#     #     fnn_df.to_csv(out_dir + 'tent_logi_fnns_r' + str(r) + '_n' + str(n) + '.csv')
#     #     print(knn_df.to_markdown())
#
#     # # ConjTest results to files
#     # for ik, k in enumerate(kv):
#     #     for it, t in enumerate(tv):
#     #         conj_df = pd.DataFrame(data=conj_diffs[:, :, ik, it], index=labels, columns=labels)
#     #         conj_df.to_csv(out_dir + 'tent_logi_conj_k' + str(k) + '_t' + str(t) + '_n' + str(n) + '.csv')
#     #         print(conj_df.to_markdown())


def generate_lorenz_data_starting_test(n=5000, starting_points=None, delay=5, emb=True, emb_dim=3):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)
    data = {}
    for idx, sp in enumerate(starting_points[:]):
        lorenz = dy.lorenz_attractor(n, starting_point=sp, skip=2000)
        if idx == 0:
            data[("lorenz", tuple(sp))] = lorenz
        if emb:
            data[("emb", tuple(sp), 0, emb_dim)] = dy.embedding(lorenz, emb_dim, dl)
    return data


def generate_lorenz_projection_test(n=5000, starting_points=None, delay=5):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)
    data = {}
    for sp in starting_points[:]:
        lorenz = dy.lorenz_attractor(n, starting_point=sp)
        data[("lorenz", tuple(sp))] = lorenz
    sp = starting_points[0]
    lorenz = data[("lorenz", tuple(sp))]
    # projections onto x-coordinate
    lx = lorenz[:, 0].reshape((n, 1))
    lz = lorenz[:, 2].reshape((n, 1))
    data[("proj", tuple(sp), 0)] = lx
    data[("proj", tuple(sp), 2)] = lz
    return data


def generate_lorenz_data_embeddings_test(n=5000, starting_points=None, delay=5, dims=[3], axis=[0]):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)

    # # projections onto x,y,z-coordinate
    # lx = lorenz[:, 0].reshape((n, 1))
    # ly = lorenz[:, 1].reshape((n, 1))
    # lz = lorenz[:, 2].reshape((n, 1))

    data = {}
    # sp = starting_points[0]
    for sp in starting_points:
        lorenz = dy.lorenz_attractor(n, starting_point=sp)
        data[("lorenz", tuple(sp))] = lorenz
        for a in axis:
            for d in dims:
                emb = dy.embedding(lorenz[:, a].reshape((n,)), d, dl)
                # if d == 1:
                #     data[('emb', tuple(sp), a, d)] = emb.reshape((len(emb),))
                # else:
                data[('emb', tuple(sp), a, d)] = emb
    # embeddings
    # rlx2 = dy.embedding(lx, 2, dl)[:, :, 0]
    # rlx3 = dy.embedding(lx, 3, dl)[:, :, 0]
    # rlx4 = dy.embedding(lx, 4, dl)[:, :, 0]
    # rly = dy.embedding(ly, 3, dl)[:, :, 0]
    # rlz = dy.embedding(lz, 3, dl)[:, :, 0]
    # data = {
    #     ("lorenz", tuple(sp)): lorenz,
    #     # ([type], [starting_points], [coordinate], [dimension])
    #     # ("emb", tuple(sp), 0, 4): rlx4,
    #     ("emb", tuple(sp), 0, 3): rlx3,
    #     # ("emb", tuple(sp), 0, 2): rlx2,
    #     # ("emb", tuple(sp), 1, 3): rly,
    #     ("emb", tuple(sp), 2, 3): rlz,
    # }
    return data


def lorenz_homeomorphisms(data, dl=5):
    # get original lorenzes as a refference sequences
    lorenzes = {}
    for l in data:
        if l[0] == 'lorenz':
            lorenzes[l[1]] = data[l]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == 'lorenz' and k2[0] == 'emb':
            refference_sequence = dy.embedding(ts1[:, k2[2]], k2[3], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'lorenz':
            refference_sequence = lorenzes[k1[1]]

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'emb':
            refference_sequence = dy.embedding(lorenzes[k1[1]][:, k2[2]], k2[3], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'lorenz' and k2[0] == 'lorenz':
            return id

    return homeo


def experiment_lorenz(test_idx=1):
    np.random.seed(0)
    n = 10000
    # delay of an embedding
    dl = 5

    tests = ["projection", "embedding", "starting_point"]
    current_test = tests[test_idx]
    pairs = None
    if current_test == "projection":
        base_name = 'lorenz_proj'
        starting_points = [[1., 1., 1.], [2., 1., 1.]]
        data = generate_lorenz_projection_test(n, starting_points, dl)
    elif current_test == "embedding":
        base_name = 'lorenz_emb'
        starting_points = [[1., 1., 1.], [2., 1., 1.]]
        data = generate_lorenz_data_embeddings_test(n, starting_points, dl, dims=[1, 2, 3], axis=[0, 2])
        pairs = [(0, i) for i in range(1, len(data.keys()))]
        # pairs = [(0, i) for i in range(9, len(data.keys()))]
        base_name = 'lorenz_emb_dmax'
    elif current_test == "starting_point":
        base_name = 'lorenz_start'
        starting_points = [[1., 1., 1.], [1.1, 1., 1.]]
        data = generate_lorenz_data_starting_test(n, starting_points, dl)

    # kv = [3,8]
    # tv = [1, 15]
    # rv = [2]
    kv = [1, 3, 5]
    tv = [1, 5, 10]
    rv = [1, 2, 3]
    experiment(data, base_name + '_n' + str(n), kv, tv, rv, True, True, lorenz_homeomorphisms(data, dl), pairs=pairs, dist_fun='max')


def generate_circle_rotation_data_test(n=1000, starting_points=None, rotations=None, nonlin_params=None):
    """
    @param n:
    @param starting_points:
    @param rotations: a list of rotations to consider (in radians)
    """
    if starting_points is None:
        starting_points = np.array([0.])
    if rotations is None:
        rotations = [0.1]

    data = {}
    for sp in starting_points:
        for r in rotations:
            print(sp, f_label(r))
            crc_points = dy.circle_rotation_interval(n, step=r, starting_point=sp)
            print(crc_points.shape)
            data[(f_label(sp), f_label(r))] = crc_points
    if nonlin_params is not None:
        for s in nonlin_params:
            sp = starting_points[0]
            r = rotations[0]
            data[(f_label(sp), f_label(r), s)] = dy.circle_rotation_interval(n, step=r, starting_point=sp, nonlin=s)
    return data


def experiment_rotation():
    n = 2000
    # starting_points = np.array([[1., 0.], [1., 0.02], [0., 1.]])
    # starting_points = np.array([[1., 0.], [0., 1.]])
    starting_points = np.array([0., 0.25])
    # starting_points = np.array([0.])
    # norms = np.linalg.norm(starting_points, axis=1)
    # starting_points = starting_points / norms[:, None]
    # rotations = [0.1, 0.11, 0.2]
    rotations = [np.sqrt(2) / 10., (np.sqrt(2) + 0.1) / 10., (np.sqrt(2) + 0.2) / 10., (np.sqrt(2)) / 5.]
    # rotations = [np.sqrt(2)/10.]
    data = generate_circle_rotation_data_test(n=n, starting_points=starting_points,
                                              rotations=rotations, nonlin_params=[5., 2.])

    # last_key = list(data.keys())[-1]
    # print(last_key)
    # for pl in [0.01, 0.02, 0.05]:
    #     np.random.seed(0)
    #     data[last_key + (str(np.round(pl, 2)),)] = perturbation_on_circle(data[last_key], pl)
    print(data.keys())
    pert_key = ('0.0', '0.141', 2.0)
    data[pert_key + ('0.05',)] = perturbation_on_circle_interval(data[pert_key], 0.05)
    pairs = [(0, i) for i in range(1, len(data.keys()))]

    def homeo(k1, k2, ts1, ts2):
        if len(k1) == len(k2):
            return id
        elif len(k1) == 2 and len(k2) > 2:
            return lambda x: angle_power_interval(x, 1. / k2[2])
        elif len(k1) > 2 and len(k2) == 2:
            return lambda x: angle_power_interval(x, k1[2])
            # return lambda x: np.power(x, 1./k1[2])

    print(sphere_max_dist(np.array([0.7]), np.array([0.3])))
    print(sphere_max_dist(np.array([0.1]), np.array([0.8])))
    # kv = [3,8]
    # tv = [1, 15]
    # rv = [2]
    kv = [1, 3, 5]
    tv = [1, 3, 5, 10]
    rv = [1, 2, 3]
    experiment(data, 'rotation_int_n' + str(n), kv, tv, rv, True, False, homeo, pairs=pairs, dist_fun=sphere_max_dist)


def generate_torus_rotation_data_test(n=2000, starting_points=None, rotations=None):
    """
    @param n:
    @param starting_points:
    @param rotations:
    """
    if starting_points is None:
        starting_points = np.array([0., 0.])
    if rotations is None:
        rotations = np.array([[0.01, 0.01]])

    data = {}
    for isp, sp in enumerate(starting_points):
        for rot in rotations:
            print(sp, f_label(rot))
            points = dy.torus_rotation_interval(n, steps=rot, starting_point=sp)
            data[('torus_rot', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points
            # if isp == 0:
            data[('torus_proj_x', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points[:, 0]
            # data[('torus_proj_y', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points[:, 1]
    return data


def experiment_torus():
    n = 2000
    # starting_points = np.array([[0., 0.]])
    starting_points = np.array([[0., 0.], [0.1, 0.]])

    # rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(2)) / 5.], [(np.sqrt(2)) / 10., (np.sqrt(3)) / 10.]])
    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.], [(1.1 * np.sqrt(2)) / 10., (np.sqrt(3)) / 10.], [np.sqrt(3) / 10., (np.sqrt(3)) / 10.]])
    # rotations = np.array([[(np.sqrt(2)) / 10., (np.sqrt(3)) / 10.], [np.sqrt(3) / 10., (np.sqrt(3)) / 10.], [np.sqrt(2) / 5., (np.sqrt(3)) / 5.]])
    data = generate_torus_rotation_data_test(n=n, starting_points=starting_points,
                                             rotations=rotations)

    pairs = [(0, i) for i in range(1, len(data.keys()))]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == k2[0]:
            return id
        elif k1[0] == 'torus_rot' and k2[0] == 'torus_proj_x':
            return lambda x: x[:, 0]
        elif k1[0] == 'torus_rot' and k1[0] == 'torus_proj_y':
            return lambda x: x[:, 1]
        elif k2[0] == 'torus_rot' and k1[0] == 'torus_proj_x':
            return lambda x: np.hstack((x.reshape(len(x),1), np.zeros((len(x), 1))))
        elif k2[0] == 'torus_rot' and k1[0] == 'torus_proj_y':
            return lambda x: np.hstack((x.reshape(len(x), 1), np.zeros((len(x), 1))))
        else:
            return None

    @numba.jit(nopython=True, fastmath=True)
    def torus_max_dist(x, y):
        dist = 0.
        modx = np.mod(x-y, 1.)
        mody = np.mod(y-x, 1.)
        for i in range(len(x)):
            di = min(modx[i], mody[i])
            dist = max(di, dist)
        return dist

    # @numba.jit(nopython=True, fastmath=True)
    # def torus_max_dist(x, y):
    #     return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=0)
        # if len(x.shape) == 1:
        #     return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=0)
        # elif len(x.shape) == 2:
        #     return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=1)
        # return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=1)
        # return np.min(np.array([np.mod(x-y, 1.), np.mod(y-x, 1.)]))
    # print(torus_max_dist(np.array([0.4, 0.3]), np.array([0.0, 0.8])))
    # print(torus_max_dist(np.array([0.4, 0.3]), np.array([0.5, 0.1])))

    kv = [1, 3, 5]
    tv = [1, 3, 5, 10]
    rv = [1, 2, 3]
    experiment(data, 'torus2_n' + str(n), kv, tv, rv, False, True, homeo, pairs=pairs, dist_fun=torus_max_dist)


def experiment(data, base_name, kv, tv, rv, do_knn, do_conj, homeo=None, pairs=None, dist_fun=None):
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())

    knn_diffs = np.ones((len(data), len(data), len(kv))) * np.infty
    fnn_diffs = np.ones((len(data), len(data), len(rv))) * np.infty
    conj_diffs = np.ones((len(data), len(data), len(kv), len(tv))) * np.infty
    neigh_conj_diffs = np.ones((len(data), len(data), len(kv), len(tv))) * np.infty

    if pairs is None:
        pairs = combinations(range(len(data)), 2)

    for (i, j) in pairs:
        k1 = keys[i]
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        if do_knn:
            knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun=dist_fun)
            knn_diffs[i, j, :] = knn1
            knn_diffs[j, i, :] = knn2

            fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun=dist_fun)
            fnn_diffs[i, j, :] = fnn1
            fnn_diffs[j, i, :] = fnn2

        # if do_conj and k1[1] != k2[1]:
        if do_conj:
            tsA = ts1[:new_n]
            tsB = ts2[:new_n]
            conj_diffs[i, j, :, :] = dy.conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=tv, dist_fun=dist_fun)
            conj_diffs[j, i, :, :] = dy.conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=tv, dist_fun=dist_fun)
            neigh_conj_diffs[i, j, :, :] = dy.neigh_conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=tv, dist_fun=dist_fun)
            neigh_conj_diffs[j, i, :, :] = dy.neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=tv, dist_fun=dist_fun)

    if do_knn:
        for ik, k in enumerate(kv):
            knn_df = pd.DataFrame(data=knn_diffs[:, :, ik], index=labels, columns=labels)
            knn_df.to_csv(out_dir + base_name + '_knns_k' + str(k) + '.csv')
            print(knn_df.to_markdown())

        for ir, r in enumerate(rv):
            fnn_df = pd.DataFrame(data=fnn_diffs[:, :, ir], index=labels, columns=labels)
            fnn_df.to_csv(out_dir + base_name + '_fnns_r' + str(r) + '.csv')
            print(knn_df.to_markdown())

    if do_conj:
        for ik, k in enumerate(kv):
            for it, t in enumerate(tv):
                conj_df = pd.DataFrame(data=conj_diffs[:, :, ik, it], index=labels, columns=labels)
                conj_df.to_csv(out_dir + base_name + '_conj_k' + str(k) + '_t' + str(t) + '.csv')
                print('conj k:' + str(k) + ' t:' + str(t))
                print(conj_df.to_markdown())
                neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[:, :, ik, it], index=labels, columns=labels)
                neigh_conj_df.to_csv(out_dir + base_name + '_neigh_conj_k' + str(k) + '_t' + str(t) + '.csv')
                print('neigh conj k:' + str(k) + ' t:' + str(t))
                print(neigh_conj_df.to_markdown())


def experiment_rotation_noise_grid():
    npoints = 2000
    base_angle = np.sqrt(2) / 10
    power_param = 2.
    data = generate_circle_rotation_data_test(n=2000, rotations=[base_angle], nonlin_params=[power_param])
    kv = [1, 3, 5, 10]
    rv = [2, 3]
    tv = [1, 3, 5, 10]
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]

    perturbation_levels = np.arange(0.00, 0.2501, 0.01)
    nkey = keys[-1]
    nnkey = nkey + (f_label(0.00),)
    data[nnkey] = data[nkey]
    data.pop(nkey)
    for ip, p, in enumerate(perturbation_levels):
        np.random.seed(0)
        data[nkey + (f_label(p),)] = perturbation_on_circle_interval(data[nnkey], p)

    keys = [k for k in data.keys()]
    column_labels = [k[-1] for k in keys[1:]]

    conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys) - 1, len(kv), 2))
    fnns_diffs = np.zeros((len(keys) - 1, len(rv), 2))

    def homeo(x):
        return angle_power_interval(x, 1. / power_param)

    k1 = keys[0]
    base_name = 'rotation_grid_dmax_noise'
    for j in range(1, len(keys)):
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        conj_diffs[j - 1, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo, k=kv, t=tv, dist_fun=sphere_max_dist)
        knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun=sphere_max_dist)
        knns_diffs[j - 1, :, 0] = knn1
        knns_diffs[j - 1, :, 1] = knn2
        fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun=sphere_max_dist)
        fnns_diffs[j - 1, :, 0] = fnn1
        fnns_diffs[j - 1, :, 1] = fnn2

    for it, t in enumerate(tv):
        conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
                               columns=column_labels)
        strt = str(t) if t > 9 else '0' + str(t)
        conj_df.to_csv(out_dir + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
        print(conj_df.to_markdown())

    knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
                            columns=column_labels)
    knns1_df.to_csv(out_dir + base_name + '_knn1' + '_n' + str(npoints) + '.csv')
    knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
                            columns=column_labels)
    knns2_df.to_csv(out_dir + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
    print(knns1_df.to_markdown())
    print(knns2_df.to_markdown())

    fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
                            columns=column_labels)
    fnns1_df.to_csv(out_dir + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
    fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
                            columns=column_labels)
    fnns2_df.to_csv(out_dir + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')


# def experiment_rotation_rv_grid():
#     # rotations = np.arange(0.05, 0.751, 0.025)
#     npoints = 2000
#     base_angle = np.sqrt(2) / 10
#     nsteps = 20
#     step = base_angle / (2 * nsteps)
#     rotations = [base_angle + (step * (i - nsteps)) for i in range(int(nsteps * 3.5))]
#     print(np.round(rotations, 4))
#     data = generate_circle_rotation_data_test(n=2000, rotations=rotations)
#     kv = [1, 3, 5, 10]
#     rv = [1, 2, 3]
#     # tv = list(range(1, 20, 5))
#     # tv = [1, 3, 5, 10, 15, 20]
#     tv = [1, 3, 5, 10]
#     keys = [k for k in data.keys()]
#     labels = [str(k) for k in keys]
#
#     conj_diffs = np.zeros((len(keys), len(kv), len(tv)))
#     knns_diffs = np.zeros((len(keys), len(kv), 2))
#     fnns_diffs = np.zeros((len(keys), len(rv), 2))
#
#     k1 = keys[int(nsteps)]
#     base_name = 'rotation_grid_r'
#     for j in range(0, len(keys)):
#         k2 = keys[j]
#         print(k1, k2)
#         if len(data[k1].shape) == 1:
#             ts1 = data[k1].reshape((len(data[k1]), 1))
#         else:
#             ts1 = data[k1]
#         if len(data[k2].shape) == 1:
#             ts2 = data[k2].reshape((len(data[k2]), 1))
#         else:
#             ts2 = data[k2]
#         new_n = min(len(ts1), len(ts2))
#         conj_diffs[j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], id, k=kv, t=tv)
#         knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
#         knns_diffs[j, :, 0] = knn1
#         knns_diffs[j, :, 1] = knn2
#         fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv)
#         fnns_diffs[j, :, 0] = fnn1
#         fnns_diffs[j, :, 1] = fnn2
#
#     for it, t in enumerate(tv):
#         conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
#                                columns=[f_label(r) for r in rotations])
#         strt = str(t) if t > 9 else '0' + str(t)
#         conj_df.to_csv(out_dir + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
#         print(conj_df.to_markdown())
#
#     knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
#                             columns=[f_label(r) for r in rotations])
#     knns1_df.to_csv(out_dir + base_name + '_knn1' + '_n' + str(npoints) + '.csv')
#     knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
#                             columns=[f_label(r) for r in rotations])
#     knns2_df.to_csv(out_dir + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
#     print(knns1_df.to_markdown())
#     print(knns2_df.to_markdown())
#
#     fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
#                             columns=[f_label(r) for r in rotations])
#     fnns1_df.to_csv(out_dir + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
#     fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
#                             columns=[f_label(r) for r in rotations])
#     fnns2_df.to_csv(out_dir + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')


def experiment_rotation_int_rv_grid():
    # rotations = np.arange(0.05, 0.751, 0.025)
    npoints = 2000
    base_angle = np.sqrt(2) / 10
    nsteps = 50
    step = base_angle / (2 * nsteps)
    rotations = [base_angle + (step * (i - nsteps)) for i in range(int(nsteps * 3.5))]
    total_steps = nsteps * 3.5
    print(np.round(rotations, 4))
    data = generate_circle_rotation_data_test(n=2000, rotations=rotations)
    kv = [1, 3, 5, 10]
    rv = [1, 2, 3]
    # tv = list(range(1, 20, 5))
    # tv = [1, 3, 5, 10, 15, 20]
    tv = [1, 3, 5, 10]
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]

    conj_diffs = np.zeros((len(keys), len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys), len(kv), 2))
    fnns_diffs = np.zeros((len(keys), len(rv), 2))

    k1 = keys[int(nsteps)]
    base_name = 'rotation_int_grid_' + str(int(total_steps)) + '_r'
    for j in range(0, len(keys)):
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        conj_diffs[j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], id, k=kv, t=tv, dist_fun=sphere_max_dist)
        knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun=sphere_max_dist)
        knns_diffs[j, :, 0] = knn1
        knns_diffs[j, :, 1] = knn2
        fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun=sphere_max_dist)
        fnns_diffs[j, :, 0] = fnn1
        fnns_diffs[j, :, 1] = fnn2

    for it, t in enumerate(tv):
        conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
                               columns=[f_label(r) for r in rotations])
        strt = str(t) if t > 9 else '0' + str(t)
        conj_df.to_csv(out_dir + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
        print(conj_df.to_markdown())

    knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in rotations])
    knns1_df.to_csv(out_dir + base_name + '_knn1' + '_n' + str(npoints) + '.csv')
    knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in rotations])
    knns2_df.to_csv(out_dir + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
    print(knns1_df.to_markdown())
    print(knns2_df.to_markdown())

    fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in rotations])
    fnns1_df.to_csv(out_dir + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
    fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in rotations])
    fnns2_df.to_csv(out_dir + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')


def experiment_lorenz_diff_sp_grid():
    data = generate_lorenz_data_starting_test(n=10000,
                                              starting_points=np.array([[1., 1., 1.], [1.01, 1., 1.], [2., 1., 1.]]),
                                              emb_dim=2)
    kv = [5, 10]
    tv = list(range(100))
    tv = list(np.arange(0, 151, 5))
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]

    conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
    neigh_conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))

    homeo = lorenz_homeomorphisms(data, dl=5)


    k1 = keys[0]
    ts1 = data[k1]
    for j in range(1, len(keys)):
        k2 = keys[j]
        if k2[0] == 'emb':
            base_name = 'lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + 'vs' + k2[0] + str(k2[3]) + str(k2[1])
        else:
            base_name = 'lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + 'vs' + k2[0] + str(k2[1])
        print(k1, k2)
        ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))

        print(base_name)
        # conj_diffs[j - 1, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv, t=tv, dist_fun=max_dist)
        # conj_df = pd.DataFrame(data=conj_diffs[j - 1, :, :], index=[str(k) for k in kv], columns=[str(t) for t in tv])
        # conj_df.to_csv(out_dir + base_name + '_conj.csv')
        # print(conj_df.to_markdown())

        neigh_conj_diffs[j - 1, :, :] = dy.neigh_conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv,
                                                                t=tv, dist_fun='max')
        neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[j - 1, :, :], index=[str(k) for k in kv],
                                     columns=[str(t) for t in tv])
        neigh_conj_df.to_csv(out_dir + base_name + '_neigh_conj.csv')
        print(neigh_conj_df.to_markdown())


def numba_dist_test():
    @numba.jit(nopython=True, fastmath=True)
    def torus_max_dist(x, y):
        dist = 0.
        modx = np.mod(x-y, 1.)
        mody = np.mod(y-x, 1.)
        for i in range(len(x)):
            di = min(modx[i], mody[i])
            dist = max(di, dist)
        return dist
    p1 = np.array([[0., 0., 0.], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    p2 = np.array([[0.3, 0.8, 0.], [0.2, 0.6, 0.15], [0.3, 0.8, 0.], [0.01, 0.1, 0.05]])
    for i in range(len(p1)):
        print(torus_max_dist(p1[i].reshape((3,)), p2[i].reshape((3,))))


if __name__ == '__main__':
    # experiment_log_tent()
    # experiment_rotation()
    # experiment_torus()
    experiment_lorenz(1)
    # experiment_rotation_tv_grid()
    # experiment_rotation_noise_grid()
    # experiment_lorenz_diff_sp_grid()
    # experiment_rotation_rv_grid()
    # experiment_rotation_int_rv_grid()
    # experiment_log_rv_grid()

    # points = np.array([[1., 0.], [0., 1.], [-1., 0.], [0, -1.]])
    # print(np.round(angle_power(points, 2.), 3))
    # numba_dist_test()
    # points = dy.torus_rotation_interval(2000, steps=np.array([np.sqrt(2)/10., np.sqrt(3)/10.]),
    #                                     starting_point=np.array([0., 0.]))
    # plt.figure()
    # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
