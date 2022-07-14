import dyrect as dy
import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import sympy

out_dir = "/Users/cybjaz/workspace/dyrect/experiments/tent_logi/"


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


def id(x):
    return x


def proj(x, i):
    return x[:, i]


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
    n = 1000

    # FNN, KNN, conjTest experiment
    # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.999, 3.99, 3.8])
    data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.99, 3.9, 3.8])
    # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0, 3.999, 3.8])
    # data = generate_log_tent_data(n, starting_points=[0.2, 0.21], log_parameters=[4.0, 3.99])
    # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0])
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())
    kv = [1, 3, 5]
    tv = [1, 3, 5]
    rv = [1, 2, 3]

    knn_diffs = np.zeros((len(data), len(data), len(kv)))
    conj_diffs = np.zeros((len(data), len(data), len(kv), len(tv)))
    fnn_diffs = np.zeros((len(data), len(data), len(rv)))

    # plt.figure()
    # x = data[keys[0]]
    # plt.plot(x)
    # plt.show()

    for (i, j) in combinations(range(len(data)), 2):
        k1 = keys[i]
        k2 = keys[j]
        print(k1, k2)
        ts1 = data[k1].reshape((n, 1))
        ts2 = data[k2].reshape((n, 1))
        new_n = min(len(ts1), len(ts2))
        knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
        knn_diffs[i, j, :] = knn1
        knn_diffs[j, i, :] = knn2

        fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv)
        fnn_diffs[i, j, :] = fnn1
        fnn_diffs[j, i, :] = fnn2

        if k1[0] == 'logm' and k2[0] == 'tent':
            h1 = h_log_to_tent
            h2 = h_tent_to_log
        elif k1[0] == 'tent' and k2[0] == 'logm':
            h1 = h_tent_to_log
            h2 = h_log_to_tent
        else:
            h1 = id
            h2 = id

        conj_diffs[i, j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], h1, k=kv, t=tv)
        conj_diffs[j, i, :, :] = dy.conjugacy_test(ts2[:new_n], ts1[:new_n], h2, k=kv, t=tv)

    for ik, k in enumerate(kv):
        knn_df = pd.DataFrame(data=knn_diffs[:, :, ik], index=labels, columns=labels)
        knn_df.to_csv(out_dir + 'tent_logi_knns_k' + str(k) + '_n' + str(n) + '.csv')
        print(knn_df.to_markdown())

    for ir, r in enumerate(rv):
        fnn_df = pd.DataFrame(data=fnn_diffs[:, :, ir], index=labels, columns=labels)
        fnn_df.to_csv(out_dir + 'tent_logi_fnns_r' + str(r) + '_n' + str(n) + '.csv')
        print(knn_df.to_markdown())

    for ik, k in enumerate(kv):
        for it, t in enumerate(tv):
            conj_df = pd.DataFrame(data=conj_diffs[:, :, ik, it], index=labels, columns=labels)
            conj_df.to_csv(out_dir + 'tent_logi_conj_k' + str(k) + '_t' + str(t) + '_n' + str(n) + '.csv')
            print(conj_df.to_markdown())


def generate_lorenz_data_starting_test(n=5000, starting_points=None, delay=5):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)
    data = {}
    for sp in starting_points[:]:
        lorenz = dy.lorenz_attractor(n, starting_point=sp)
        # projections onto x-coordinate
        lx = lorenz[:, 0].reshape((n, 1))
        rlx3 = dy.embedding(lx, 3, dl)[:, :, 0]
        data[("lorenz", tuple(sp))] = lorenz
        data[("emb", tuple(sp), 0, 3)] = rlx3
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


def generate_lorenz_data_embeddings_test(n=5000, starting_points=None, delay=5):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)

    sp = starting_points[0]
    lorenz = dy.lorenz_attractor(n, starting_point=sp)
    # projections onto x,y,z-coordinate
    lx = lorenz[:, 0].reshape((n, 1))
    ly = lorenz[:, 1].reshape((n, 1))
    lz = lorenz[:, 2].reshape((n, 1))
    # embeddings
    rlx2 = dy.embedding(lx, 2, dl)[:, :, 0]
    rlx3 = dy.embedding(lx, 3, dl)[:, :, 0]
    rlx4 = dy.embedding(lx, 4, dl)[:, :, 0]
    rly = dy.embedding(ly, 3, dl)[:, :, 0]
    rlz = dy.embedding(lz, 3, dl)[:, :, 0]
    data = {
        ("lorenz", tuple(sp)): lorenz,
        # ([type], [starting_points], [coordinate], [dimension])
        ("emb", tuple(sp), 0, 4): rlx4,
        ("emb", tuple(sp), 0, 3): rlx3,
        ("emb", tuple(sp), 0, 2): rlx2,
        # ("emb", tuple(sp), 1, 3): rly,
        ("emb", tuple(sp), 2, 3): rlz,
    }
    return data


def experiment_lorenz(test_idx=0):
    np.random.seed(0)
    n = 10000
    # delay of an embedding
    dl = 5

    do_knn_fnn = False
    do_conj = True

    # FNN, KNN, conjTest experiment
    tests = ["projection", "embedding", "starting_point"]
    current_test = tests[test_idx]
    if current_test == "projection":
        base_name = 'lorenz_proj'
        starting_points = [[1., 1., 1.], [2., 1., 1.]]
        data = generate_lorenz_projection_test(n, starting_points, dl)
    elif current_test == "embedding":
        base_name = 'lorenz_emb'
        starting_points = [[1., 1., 1.]]
        data = generate_lorenz_data_embeddings_test(n, starting_points, dl)
    elif current_test == "starting_point":
        base_name = 'lorenz_start'
        starting_points = [[1., 1., 1.], [1.1, 1., 1.]]
        data = generate_lorenz_data_starting_test(n, starting_points, dl)
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())
    kv = [1, 3, 5, 8]
    tv = [1, 3]
    # kv = [1, 3, 5]
    # tv = [1, 3, 5]
    rv = [1, 2, 3]
    knn_diffs = np.zeros((len(data), len(data), len(kv)))
    conj_diffs = np.zeros((len(data), len(data), len(kv), len(tv)))
    fnn_diffs = np.zeros((len(data), len(data), len(rv)))

    for (i, j) in combinations(range(len(data)), 2):
        k1 = keys[i]
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1][0]) == 1:
            ts1 = data[k1].reshape((n, 1))
        else:
            ts1 = data[k1]
        if len(data[k2][0]) == 1:
            ts2 = data[k2].reshape((n, 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        if do_knn_fnn:
            print("KNN")
            knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
            knn_diffs[i, j, :] = knn1
            knn_diffs[j, i, :] = knn2

            print("FNN")
            fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv)
            fnn_diffs[i, j, :] = fnn1
            fnn_diffs[j, i, :] = fnn2

        if k1[0] == 'lorenz' and k2[0] == 'proj':
            def h(x):
                return proj(x, k2[2])

            h1 = h
            h2 = None
        elif k1[0] == 'lorenz' and k2[0] == 'lorenz':
            h1 = id
            h2 = id
        elif k1[0] == 'proj' and k2[0] == 'proj' and k1[2] == k2[2]:
            h1 = id
            h2 = id
        # elif k1[0] == 'emb' and k2[0] == 'proj' and k1[2] == k2[2]:
        #     def h(x):
        #         return proj(x, 0)
        #     h1 = h
        #     h2 = None
        # elif k1[0] == 'proj' and k2[0] == 'emb' and k1[2] == k2[2]:
        #     def h(x):
        #         return proj(x, 0)
        #     h1 = None
        #     h2 = h
        elif k1[0] == 'lorenz' and k2[0] == 'emb':
            def h(x):
                image = np.zeros((len(x), k2[3]))
                for p in range(len(x)):
                    idx = np.argwhere(ts1 == x[p])[0, 0]
                    image[p, :] = np.array([ts1[idx + d * dl, k2[2]] for d in range(k2[3])])
                return image

            h1 = h
            if k2[3] == 3:
                h2 = id
            else:
                h2 = None
        elif k1[0] == 'emb' and k2[0] == 'lorenz':
            def h(x):
                image = np.zeros((len(x), k1[3]))
                for p in range(len(x)):
                    idx = np.argwhere(ts2 == x[p])[0, 0]
                    image[p, :] = np.array([ts2[idx + d * dl, k1[2]] for d in range(k1[3])])
                return image

            if k1[3] == 3:
                h1 = id
            else:
                h1 = None
            h2 = h
        # if we have an embedding of the same coordinate
        elif k1[0] == 'emb' and k2[0] == 'emb' and k1[2] == k2[2]:
            if k1[3] >= k2[3]:
                def h12(x):
                    return x[:, :k2[3]]

                h1 = h12
            else:
                h1 = None
            if k2[3] >= k1[3]:
                def h21(x):
                    return x[:, :k1[3]]

                h2 = h21
            else:
                h2 = None
        # elif k1[0] == k2[0]:
        #     h1 = id
        #     h2 = id
        else:
            h1 = None
            h2 = None

        if do_conj:
            print("conj")
            if h1 is not None:
                conj_diffs[i, j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], h1, k=kv, t=tv)
            else:
                conj_diffs[i, j, :, :] = np.infty
            if h2 is not None:
                conj_diffs[j, i, :, :] = dy.conjugacy_test(ts2[:new_n], ts1[:new_n], h2, k=kv, t=tv)
            else:
                conj_diffs[j, i, :, :] = np.infty

    if do_knn_fnn:
        for ik, k in enumerate(kv):
            knn_df = pd.DataFrame(data=knn_diffs[:, :, ik], index=labels, columns=labels)
            knn_df.to_csv(out_dir + base_name + '_knns_k' + str(k) + '_n' + str(n) + '.csv')
            print(knn_df.to_markdown())

        for ir, r in enumerate(rv):
            fnn_df = pd.DataFrame(data=fnn_diffs[:, :, ir], index=labels, columns=labels)
            fnn_df.to_csv(out_dir + base_name + '_fnns_r' + str(r) + '_n' + str(n) + '.csv')
            print(fnn_df.to_markdown())

    if do_conj:
        for ik, k in enumerate(kv):
            for it, t in enumerate(tv):
                conj_df = pd.DataFrame(data=conj_diffs[:, :, ik, it], index=labels, columns=labels)
                conj_df.to_csv(out_dir + base_name + '_conj_k' + str(k) + '_t' + str(t) + '_n' + str(n) + '.csv')
                print(conj_df.to_markdown())


def generate_circle_rotation_data_test(n=1000, starting_points=None, rotations=None):
    """
    @param n:
    @param starting_points:
    @param rotations: a list of rotations to consider (in radians)
    """
    if starting_points is None:
        starting_points = np.array([[1., 0.]])
    if rotations is None:
        rotations = [0.1]

    data = {}
    for sp in starting_points:
        for r in rotations:
            print(sp, r)
            data[(tuple(sp), r)] = dy.circle_rotation(n, step=r, starting_point=sp)
    return data


def experiment_rotation():
    n = 1000
    starting_points = np.array([[1., 0.], [1., 0.02], [0., 1.]])
    # starting_points = np.array([[1., 0.], [0., 1.]])
    norms = np.linalg.norm(starting_points, axis=1)
    starting_points = starting_points / norms[:, None]
    rotations = [0.1, 0.11, 0.2]
    # rotations = [0.1, 0.2]
    data = generate_circle_rotation_data_test(n=n, starting_points=starting_points, rotations=rotations)

    def homeo(key1, key2):
        return id

    # kv = [3,8]
    # tv = [1, 15]
    # rv = [2]
    kv = [1, 3, 5, 8]
    tv = [1, 3, 5, 15]
    rv = [1, 2, 3]
    experiment(data, 'rotation_n' + str(n), kv, tv, rv, True, True, homeo)


def experiment(data, base_name, kv, tv, rv, do_knn, do_conj, homeo=None):
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())

    knn_diffs = np.zeros((len(data), len(data), len(kv)))
    conj_diffs = np.zeros((len(data), len(data), len(kv), len(tv)))
    fnn_diffs = np.zeros((len(data), len(data), len(rv)))

    for (i, j) in combinations(range(len(data)), 2):
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
            knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
            knn_diffs[i, j, :] = knn1
            knn_diffs[j, i, :] = knn2

            fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=rv)
            fnn_diffs[i, j, :] = fnn1
            fnn_diffs[j, i, :] = fnn2

        if do_conj:
            conj_diffs[i, j, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2), k=kv, t=tv)
            conj_diffs[j, i, :, :] = dy.conjugacy_test(ts2[:new_n], ts1[:new_n], homeo(k2, k1), k=kv, t=tv)

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
                print(conj_df.to_markdown())


if __name__ == '__main__':
    # experiment_log_tent()
    experiment_rotation()
    # experiment_lorenz(2)
