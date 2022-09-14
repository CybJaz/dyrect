import dyrect as dy
from itertools import combinations
import numpy as np
import numba
import pandas as pd

out_dir = "/Users/cybjaz/workspace/dyrect/experiments/conjugacy_experiment_output/"
log_dir = 'log_map/'
rot_dir = 'circle_rotation/'
lor_dir = 'lorenz/'
kle_dir = 'klein/'
dad_dir = 'dadras/'

def f_label(x):
    return str(np.round(x, 3))


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


def embedding_homeomorphisms(data, base_ts, dl=5, emb_ts='emb'):
    # the structure of embedding time series:
    # (emb_ts, [index of projected coordinate], [embedding dimension], [other parameters of the series], [other2], ...)
    # the structure of the base time series:
    # (base_ts, [parameter of a time series], [parameter2], ...)

    # get original time series serving later as a refference sequences
    base_time_series = {}
    for l in data:
        if l[0] == base_ts:
            base_time_series[l[1:]] = data[l]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == base_ts and k2[0] == 'emb':
            refference_sequence = dy.embedding(ts1[:, k2[1]], k2[2], dl)
            # print(refference_sequence.shape)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == base_ts:
            refference_sequence = base_time_series[k1[3:]]

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'emb':
            refference_sequence = dy.embedding(base_time_series[k1[3:]][:, k2[1]], k2[2], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == base_ts and k2[0] == base_ts:
            return id

    return homeo

########################################################
############### Logistic map experiments ###############
########################################################
def generate_log_tent_data(n=500, starting_points=None, log_parameters=None):
    # starting point
    # starting_points = [sympy.Rational(20, 100), sympy.Rational(21, 100), sympy.Rational(25, 100)]
    # parameters = [sympy.Rational(4, 1)]
    if starting_points is None:
        starting_points = [0.2]
    if log_parameters is None:
        log_parameters = [4.]

    data = {}
    for p in log_parameters:
        for sp in starting_points:
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
    kv = [3, 5]
    # kv = [10]
    tv = [5, 10]
    rv = [2, 3]

    pairs = [(1, i) for i in range(0, len(data.keys()))]
    # pairs = [(1,0), (1, 4), (1, 2), (1, 5)]
    # pairs = [(1,4)]
    def homeo(k1, k2, ts1, ts2):
        if k1[0] == 'logm' and k2[0] == 'tent':
            return h_log_to_tent
        elif k1[0] == 'tent' and k2[0] == 'logm':
            return h_tent_to_log
        else:
            return id

    experiment(data, log_dir+'log_tent_n' + str(n), kv, tv, rv, True, True, homeo, pairs=pairs)


def experiment_log_rv_grid():
    npoints = 2000
    log_params = np.arange(4., 3.8, -0.005)
    total_steps = len(log_params)
    data = generate_log_tent_data(n=2000, starting_points=[0.2], log_parameters=log_params)
    kv = [5, 10]
    rv = [2, 3]
    # tv = list(range(1, 20, 5))
    # tv = [1, 3, 5, 10, 15, 20]
    tv = [1, 3, 5, 10, 20]
    data.pop(list(data.keys())[0])
    keys = [k for k in data.keys()]

    conj_diffs = np.zeros((len(keys), len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys), len(kv), 2))
    fnns_diffs = np.zeros((len(keys), len(rv), 2))

    k1 = keys[0]
    base_name = log_dir + 'log_params_grid_' + str(int(total_steps)) + '_'
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


###########################################################
############### Circle rotation experiments ###############
###########################################################


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
    starting_points = np.array([0., 0.25])
    starting_points = np.array([0.])
    rotations = [np.sqrt(2) / 10., (np.sqrt(2) + 0.2) / 10., (np.sqrt(2)) / 5.]
    data = generate_circle_rotation_data_test(n=n, starting_points=starting_points,
                                              rotations=rotations, nonlin_params=[2.])

    last_key = list(data.keys())[-1]
    print(data.keys())
    pert_key = ('0.0', '0.141', 2.0)
    data[pert_key + ('0.05',)] = perturbation_on_circle_interval(data[pert_key], 0.05)
    pairs = [(0, i) for i in range(1, len(data.keys()))]
    print(pairs)

    def homeo(k1, k2, ts1, ts2):
        if len(k1) == len(k2):
            return id
        elif len(k1) == 2 and len(k2) > 2:
            return lambda x: angle_power_interval(x, 1. / k2[2])
        elif len(k1) > 2 and len(k2) == 2:
            return lambda x: angle_power_interval(x, k1[2])

    # kv = [3,8]
    # tv = [1, 15]
    # rv = [2]
    kv = [3, 5]
    tv = [5, 10]
    rv = [2, 3]
    experiment(data, rot_dir + 'rotation_n' + str(n), kv, tv, rv, True, True, homeo, pairs=pairs,
               dist_fun=sphere_max_dist)


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
        conj_diffs[j - 1, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo, k=kv, t=tv,
                                                    dist_fun=sphere_max_dist)
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


##########################################################
############### Torus rotation experiments ###############
##########################################################


# <editor-fold desc="Torus rotation">
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
            # print(sp, f_label(rot))
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
    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.], [(1.1 * np.sqrt(2)) / 10., (np.sqrt(3)) / 10.],
                          [np.sqrt(3) / 10., (np.sqrt(3)) / 10.]])
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
            return lambda x: np.hstack((x.reshape(len(x), 1), np.zeros((len(x), 1))))
        elif k2[0] == 'torus_rot' and k1[0] == 'torus_proj_y':
            return lambda x: np.hstack((x.reshape(len(x), 1), np.zeros((len(x), 1))))
        else:
            return None

    @numba.jit(nopython=True, fastmath=True)
    def torus_max_dist(x, y):
        dist = 0.
        modx = np.mod(x - y, 1.)
        mody = np.mod(y - x, 1.)
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
# </editor-fold>


#########################################################
############### Lorenz system experiments ###############
#########################################################

def generate_lorenz_data_starting_test(n=5000, starting_points=None, delay=5, emb=True, emb_dim=3):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)
    data = {}
    for idx, sp in enumerate(starting_points[:]):
        lorenz = dy.lorenz_attractor(n, starting_point=sp, skip=2000)
        # if idx == 0:
        # A = np.array([[1 / 3., 1 / 3., 1 / 3.],
        #               [1 / 3., 1 / 3., -1 / 3.],
        #               [1 / 3., -1 / 3., -1 / 3.]])
        # lorenz = np.dot(A, lorenz.transpose()).transpose()
        data[("lorenz", tuple(sp))] = lorenz
        if emb:
            # data[("emb", tuple(sp), 0, emb_dim)] = dy.embedding(lorenz, emb_dim, dl)
            lemb = dy.embedding(lorenz[:, 0].reshape((n,)), emb_dim, dl)
            data[("emb", 0, emb_dim, tuple(sp))] = lemb
    return data


def generate_lorenz_data_embeddings_test(n=5000, starting_points=None, delay=5, dims=None, axis=None):
    if dims is None:
        dims = [3]
    if axis is None:
        axis = [0]
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)

    data = {}
    for sp in starting_points:
        lorenz = dy.lorenz_attractor(n, starting_point=sp, skip=2000)
        # A = np.array([[1 / 3., 1 / 3., 1 / 3.],
        #               [1 / 3., 1 / 3., -1 / 3.],
        #               [1 / 3., -1 / 3., -1 / 3.]])
        # lorenz = np.dot(A, lorenz.transpose()).transpose()
        data[("lorenz", tuple(sp))] = lorenz
        for a in axis:
            for d in dims:
                emb = dy.embedding(lorenz[:, a].reshape((n,)), d, dl)
                # data[('emb', tuple(sp), a, d)] = emb
                data[('emb', a, d, tuple(sp))] = emb
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


def experiment_lorenz(test_idx=0):
    np.random.seed(0)
    n = 10000
    # delay of an embedding
    dl = 5

    tests = ["embedding", "starting_point"]
    current_test = tests[test_idx]
    pairs = None
    if current_test == "embedding":
        base_name = 'lorenz_emb'
        starting_points = [[1., 1., 1.], [2., 1., 1.]]
        # data = generate_lorenz_data_embeddings_test(n, starting_points, dl, dims=[1, 2, 3], axis=[0, 2])
        # pairs = [(0, i) for i in range(1, len(data.keys()))]
        data = generate_lorenz_data_embeddings_test(n, starting_points, dl, dims=[1, 2, 3, 4], axis=[0, 2])
        pairs = [(0, i) for i in range(0, len(data.keys()))]
        # pairs = [(0, i) for i in range(1, 7)]
        # pairs = [(0, i) for i in range(7, len(data.keys()))]
        # pairs = [(0, i) for i in range(9, len(data.keys()))]
        base_name = 'lorenz_emb_dmax'
    elif current_test == "starting_point":
        base_name = 'lorenz_start'
        starting_points = [[1., 1., 1.], [1.1, 1., 1.]]
        data = generate_lorenz_data_starting_test(n, starting_points, dl)

    # kv = [3,8]
    # tv = [1, 15]
    # rv = [2]
    # kv = [3, 5]
    # tv = [5, 10]
    kv = [5]
    tv = [5, 10]
    rv = [2,3]
    experiment(data, lor_dir + base_name + '_n' + str(n), kv, tv, rv, True, True,
               embedding_homeomorphisms(data, 'lorenz', dl),
               # lorenz_homeomorphisms(data,  dl),
               pairs=pairs, dist_fun='max')


def experiment_lorenz_diff_sp_grid():
    dl = 5
    for embv in [1, 2, 3, 4]:
        data = generate_lorenz_data_starting_test(n=10000,
                                                  starting_points=np.array([[1., 1., 1.], [1., 2., 1.], [2., 1., 1.], [1., 1., 2.]]),
                                                  # starting_points=np.array([[1., 1., 1.], [1., 2., 1.], [2., 1., 1.], [1., 1., 2.]]),
                                                  delay=dl,
                                                  emb_dim=embv)
        kv = [5]
        # tv = list(range(100))
        tv = list(np.arange(1, 12, 2)) + list(np.arange(15, 26, 5)) + list(np.arange(30, 101, 10))
        keys = [k for k in data.keys()]

        conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
        neigh_conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))

        # homeo = lorenz_homeomorphisms(data, dl=5)
        homeo = embedding_homeomorphisms(data, 'lorenz', dl=dl)

        rev = True

        if rev:
            k2 = keys[0]
            ts2 = data[k2]
        else:
            k1 = keys[0]
            ts1 = data[k1]

        for j in range(2, len(keys)):
            if rev:
                k1 = keys[j]
                ts1 = data[k1]
                if k1[0] == 'emb':
                    # base_name = lor_dir + 'rev_lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[3]) + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
                    base_name = lor_dir + 'rev_lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[2]) + str(k1[3]) + '_vs_' + k2[0] + str(k2[1])
                elif embv == 2:
                    base_name = lor_dir + 'rev_lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
                else:
                    base_name = 'wut'
                    continue
            else:
                k2 = keys[j]
                ts2 = data[k2]
                if k2[0] == 'emb':
                    base_name = lor_dir + 'rev_lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[3]) + str(k2[1])
                else:
                    base_name = lor_dir + 'rev_lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
            print(k1, k2)
            # ts2 = data[k2]
            new_n = min(len(ts1), len(ts2))

            print(base_name)
            # conj_diffs[j - 1, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv, t=tv,
            #                                             dist_fun='max')
            # conj_df = pd.DataFrame(data=conj_diffs[j - 1, :, :], index=[str(k) for k in kv], columns=[str(t) for t in tv])
            # conj_df.to_csv(out_dir + base_name + '_conj.csv')
            # print(conj_df.to_markdown())

            neigh_conj_diffs[j - 1, :, :] = dy.neigh_conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv,
                                                                    t=tv, dist_fun='max')
            neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[j - 1, :, :], index=[str(k) for k in kv],
                                         columns=[str(t) for t in tv])
            neigh_conj_df.to_csv(out_dir + base_name + '_neigh_conj.csv')
            print(neigh_conj_df.to_markdown())


##########################################################
############### Klein rotation experiments ###############
##########################################################

def generate_klein_rotation_data_test(n=2000, starting_points=None, rotations=None, delay=8, dims=None):
    """
    @param n:
    @param starting_points:
    @param rotations:
    """
    if starting_points is None:
        starting_points = np.array([0., 0.])
    if dims is None:
        dims = [4]
    if rotations is None:
        rotations = np.array([[0.01, 0.01]])

    # parameters of the klein bottle
    kr = 1.
    kp = 8.
    ke = 1. / 2.

    data = {}
    for isp, sp in enumerate(starting_points):
        for rot in rotations:
            # print(sp, f_label(rot))
            tor = dy.torus_rotation_interval(n, steps=rot, starting_point=sp) * 2 * np.pi
            A = np.array([[1 / 4., 1 / 4., 1 / 4., 1 / 4.], [1 / 4., 1 / 4., 1 / 4., -1 / 4.],
                          [1 / 4., 1 / 4., -1 / 4., -1 / 4.], [1 / 4., -1 / 4., -1 / 4., -1 / 4.]])
            cos0d2 = np.cos(tor[:, 0] / 2.)
            sin0d2 = np.sin(tor[:, 0] / 2.)
            cos0 = np.cos(tor[:, 0])
            sin0 = np.sin(tor[:, 0])
            sin1 = np.sin(tor[:, 1])
            cos1 = np.cos(tor[:, 1])
            sin1m2 = np.sin(tor[:, 1] * 2.)
            klein = np.array([kr * (cos0d2 * cos1 - sin0d2 * sin1m2),
                              kr * (sin0d2 * cos1 - cos0d2 * sin1m2),
                              kp * cos0 * (1 + ke * sin1),
                              kp * sin0 * (1 + ke * sin1)]).transpose().reshape((len(tor), 4))
            shifted_klein = np.dot(A, klein.transpose()).transpose()
            data[('klein', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = klein
            for d in dims:
                emb = dy.embedding(shifted_klein[:, 0].reshape((n,)), d, delay)
                data[('emb', 0, d, f_label(sp), f_label(rot[0]), f_label(rot[1]))] = emb
    return data

def experiment_klein_diff_sp_grid():
    dl = 8

    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.]])
    for embv in [2, 3, 4, 5]:
        data = generate_klein_rotation_data_test(n=8000,
                                                  rotations=rotations,
                                                  starting_points=np.array([[0., 0.], [0.3, 0.1], [0.2, 0.25], [0.1, 0.4]]),
                                                  delay=dl,
                                                  dims=[embv])
        kv = [5]
        # tv = list(range(100))
        tv = list(np.arange(1, 11, 2))  + [12, 15, 20, 25, 30, 40, 50] # +  list(np.arange(30, 81, 10))
        keys = [k for k in data.keys()]

        conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
        neigh_conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
        homeo = embedding_homeomorphisms(data, 'klein', dl=dl)

        rev = True

        if rev:
            k2 = keys[0]
            ts2 = data[k2]
        else:
            k1 = keys[0]
            ts1 = data[k1]

        for j in range(2, len(keys)):
            if rev:
                k1 = keys[j]
                ts1 = data[k1]
                if k1[0] == 'emb':
                    base_name = kle_dir + 'rev_klein_stpts_grid_dmax_t_' + k1[0] + str(k1[2]) + str(k1[3]) + '_vs_' + k2[0] + str(k2[1])
                elif embv == 2 and k1[0] == 'klein':
                    base_name = kle_dir + 'rev_klein_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
                else:
                    base_name = 'what?'
                    continue
            else:
                k2 = keys[j]
                ts2 = data[k2]
                if k2[0] == 'emb':
                    base_name = kle_dir + 'klein_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[3]) + str(k2[1])
                else:
                    base_name = kle_dir + 'klein_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
            print(k1, k2)
            # ts2 = data[k2]
            new_n = min(len(ts1), len(ts2))

            print(base_name)
            # conj_diffs[j - 1, :, :] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv, t=tv,
            #                                             dist_fun='max')
            # conj_df = pd.DataFrame(data=conj_diffs[j - 1, :, :], index=[str(k) for k in kv], columns=[str(t) for t in tv])
            # conj_df.to_csv(out_dir + base_name + '_conj.csv')
            # print(conj_df.to_markdown())

            neigh_conj_diffs[j - 1, :, :] = dy.neigh_conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv,
                                                                    t=tv, dist_fun='max')
            neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[j - 1, :, :], index=[str(k) for k in kv],
                                         columns=[str(t) for t in tv])
            neigh_conj_df.to_csv(out_dir + base_name + '_neigh_conj.csv')
            print(neigh_conj_df.to_markdown())

#########################################################
############### Dadras system experiments ###############
#########################################################

def dadras_homeomorphisms(data, dl=5):
    # get original dadrases as a refference sequences
    dadrases = {}
    for l in data:
        if l[0] == 'dadras':
            dadrases[l[1]] = data[l]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == 'dadras' and k2[0] == 'emb':
            refference_sequence = dy.embedding(ts1[:, k2[2]], k2[3], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'dadras':
            refference_sequence = dadrases[k1[1]]

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'emb':
            refference_sequence = dy.embedding(dadrases[k1[1]][:, k2[2]], k2[3], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'dadras' and k2[0] == 'dadras':
            return id

    return homeo


def generate_dadras_data_embeddings_test(n=10000, starting_points=None, delay=8, dims=None, axis=None):
    if dims is None:
        dims = [3]
    if axis is None:
        axis = [0]
    if starting_points is None:
        starting_points = [[10., 1., 10., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)

    data = {}
    for sp in starting_points:
        dadras = dy.dadras_attractor(n, starting_point=sp)
        # A = np.array([[1 / 4., 1 / 4., 1 / 4., 1 / 4.], [1 / 4., 1 / 4., 1 / 4., -1 / 4.],
        #               [1 / 4., 1 / 4., -1 / 4., -1 / 4.], [1 / 4., -1 / 4., -1 / 4., -1 / 4.]])
        # dadras = np.dot(A, dadras.transpose()).transpose()
        data[("dadras", tuple(sp))] = dadras
        for a in axis:
            for d in dims:
                emb = dy.embedding(dadras[:, a].reshape((n,)), d, dl)
                data[('emb', a, d, tuple(sp))] = emb
    return data


def experiment_dadras():
    np.random.seed(0)
    n = 10000
    # delay of an embedding
    dlv = [10]


    for dl in dlv:
        pairs = None
        starting_points = [[10., 1., 10., 1.], [1., 10., 10., 1.]]
        data = generate_dadras_data_embeddings_test(n, starting_points, dl, dims=[1, 2, 3, 4], axis=[1])
        # pairs = [(0, i) for i in range(1, len(data.keys()))]
        pairs = [(0, i) for i in range(5, len(data.keys()))]
        # pairs = [(0, 5), (1, 6), (1, 7), (2, 8), (3, 9)]
        pairs = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]
        base_name = 'dadras_emb' + '_dl' + str(dl)

        # kv = [3,8]
        # tv = [1, 15]
        # rv = [2]
        kv = [5]
        tv = [10]
        rv = [2, 3]
        print(base_name)
        experiment(data, dad_dir + base_name + '_n' + str(n), kv, tv, rv, False, True,
                   embedding_homeomorphisms(data, 'dadras', dl=dl, emb_ts='emb'), pairs=pairs,
                   dist_fun='max')

###############################################
############### General methods ###############
###############################################

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
        print(k1, ' vs. ', k2)
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
            neigh_conj_diffs[i, j, :, :] = dy.neigh_conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=tv,
                                                                   dist_fun=dist_fun)
            neigh_conj_diffs[j, i, :, :] = dy.neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=tv,
                                                                   dist_fun=dist_fun)

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


def numba_dist_test():
    @numba.jit(nopython=True, fastmath=True)
    def torus_max_dist(x, y):
        dist = 0.
        modx = np.mod(x - y, 1.)
        mody = np.mod(y - x, 1.)
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
    # experiment_log_rv_grid()

    # experiment_rotation()
    # experiment_rotation_noise_grid()
    # experiment_rotation_int_rv_grid()

    # experiment_torus()

    experiment_lorenz(0)
    # experiment_lorenz_diff_sp_grid()

    # experiment_klein_diff_sp_grid()
    # experiment_dadras()

    # points = np.array([[1., 0.], [0., 1.], [-1., 0.], [0, -1.]])
    # print(np.round(angle_power(points, 2.), 3))
    # numba_dist_test()
    # points = dy.torus_rotation_interval(2000, steps=np.array([np.sqrt(2)/10., np.sqrt(3)/10.]),
    #                                     starting_point=np.array([0., 0.]))
    # plt.figure()
    # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
