import dyrect as dy
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

def short_test():
    datapath = '/home/cybjaz/workspace/DyRectLibrary/data/time_series/'
    time_series = []
    for i in range(1, 4):
        with open(datapath + 'trajectory' + str(i), newline='') as csvfile:
            ts = np.loadtxt(csvfile, delimiter='\t')
            time_series.append(ts)
    print(np.array(time_series[0]).shape)

    print(dy.conjugacy_test_knn(time_series[0], time_series[1], k=5))
    print(dy.conjugacy_test_knn(time_series[0], time_series[2], k=5))
    print(dy.conjugacy_test_knn(time_series[1], time_series[2], k=5))


def lorenz_test():
    n = 10000
    k = 10
    dl = 5
    np.random.seed(0)
    lorenz = dy.lorenz_attractor(n)
    # print(lorenz.shape)
    print(lorenz[:, 0].reshape((n, 1)).shape)
    lx = lorenz[:, 0].reshape((n, 1))
    ly = lorenz[:, 1].reshape((n, 1))
    lz = lorenz[:, 2].reshape((n, 1))

    rlx = dy.embedding(lx, 3, dl)[:,:,0]
    rly = dy.embedding(ly, 3, dl)[:,:,0]
    rlz = dy.embedding(lz, 3, dl)[:,:,0]

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # ax.scatter(rlx[:,0], rlx[:,1], rlx[:,2], s=0.1)
    # ax = fig.add_subplot(2, 2, 2, projection='3d')
    # ax.scatter(rly[:,0], rly[:,1], rly[:,2], s=0.1)
    # ax = fig.add_subplot(2, 2, 3, projection='3d')
    # ax.scatter(rlz[:,0], rlz[:,1], rlz[:,2], s=0.1)
    # plt.show()

    # ts1 = rlx
    # ts2 = rlz
    #
    # new_n = min(len(ts1), len(ts2))
    # ts1 = ts1[:new_n]
    # ts2 = ts2[:new_n]
    # d, c = dy.conjugacy_knn(ts1, ts2, k, point_vals=True)

    ## Draw conjugacy error
    # max_v = np.max([np.max(c[0]), np.max(c[1])])
    # print(d)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(ts1[:,0], ts1[:,1], ts1[:,2], c=np.array(c[0])/max_v, s=0.6)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(ts2[:,0], ts2[:,1], ts2[:,2], c=np.array(c[1])/max_v, s=0.6)
    # # ax.scatter(lorenz[:new_n, 0], lorenz[:new_n, 1], lorenz[:new_n, 2], c=np.array(c[1]) / max_v, s=0.6)
    # plt.show()


    # ### KNN for lorenz
    # to_compare = [lorenz, rlx, rly, rlz]
    # labels = ["lorenz", "rlx", "rly", "rlz"]
    #
    # k_values = [1, 4, 8, 12]
    #
    # knns = np.zeros((len(to_compare), len(to_compare), len(k_values)))
    # # knns2 = np.zeros((len(to_compare), len(to_compare), len(k_values)))
    # for (i, j) in combinations(range(len(to_compare)), 2):
    #     print(labels[i] + " vs. " + labels[j])
    #     ts1 = to_compare[i]
    #     ts2 = to_compare[j]
    #     new_n = min(len(ts1), len(ts2))
    #     knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=k_values)
    #     knns[i, j, :] = knn1
    #     knns[j, i, :] = knn2
    #
    # for ki, kv in enumerate(k_values):
    #     knn_df = pd.DataFrame(data=knns[:,:,ki],  index=labels, columns=labels)
    #     # knn_df.name
    #     print("K = " + str(kv))
    #     print(knn_df)
    #     print(" ")
    #     np.savetxt('lorenz_knns_k'+str(kv)+'.csv', knns[:,:,ki], delimiter=",")

    ### FNN for lorenz
    to_compare = [lorenz, rlx, rly, rlz]
    labels = ["lorenz", "rlx", "rly", "rlz"]

    for (i, j) in combinations(range(len(to_compare)), 2):
        print(labels[i] + " vs. " + labels[j])
        ts1 = to_compare[i]
        ts2 = to_compare[j]
        new_n = min(len(ts1), len(ts2))
        fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=range(1,8))
        plt.figure()
        plt.plot(fnn1, label=labels[i] + " to " + labels[j])
        plt.plot(fnn2, label=labels[j] + " to " + labels[i])
        plt.legend()
    plt.show()

def fnn_test():
    # n = 200
    # # points1 = dy.logistic_map(n, starting_point=0.1)
    # # points2 = np.random.random((n,))
    # # points1 = np.linspace(0, 1, n)
    # # print(points)
    # sin_points = np.array([np.sin(np.sqrt(2) * pi * i / 100) for i in range(n)])
    # points = [dy.embedding(sin_points, d, 20) for d in range(1, 6)]
    # for p in points:
    #     # print(p[:5,:])
    #     print(p.shape)

    n = 1000
    # k = 10
    # dl = 5
    np.random.seed(0)
    lorenz = dy.lorenz_attractor(n)[:,0]
    points = [dy.embedding(lorenz, d, 4) for d in range(1, 6)]

    n = min([len(p) for p in points])

    fnns = []
    for i in range(len(points) - 1):
        fnns.append([dy.fnn(points[i][:n], points[i+1][:n], r=r)[0] for r in range(1,12)])
        print(fnns[-1])

    plt.figure()
    i = 1
    for fnn in fnns:
        plt.plot(fnn, label=str(i))
        i += 1
    plt.legend()

    # plt.figure()
    # plt.scatter(points[1][:,0], points[1][:,1])
    plt.show()
    # dy.fnn(points1, points2)


def logi_tent_test():
    import sympy
    n = 1000
    # sp = sympy.Rational(1, 5)
    sp = 0.2
    # sp = np.sqrt(5) / 5.
    # sp = sympy.sqrt(3)/2

    def homeo(x):
        return 2 * sympy.asin(sympy.sqrt(x)) / sympy.pi
    def homeo1(x):
        return sympy.sin(sympy.pi * x / 2)**2
    # sp = sympy.Rational(7, 9)
    print(homeo1(sp))

    # tent = dy.tent_map(n, starting_point=sp)
    # logi = np.array([homeo1(x).evalf() for x in tent], dtype=float)

    logi = np.array(dy.logistic_map(n, starting_point=sp))
    # tent = np.array(dy.logistic_map(n, r=4.00, starting_point=sp))
    tent = np.array([homeo(x).evalf() for x in logi], dtype=float)
    # tent = np.linspace(0,1., n)

    # def t(x):
    #     return 2 * x if x<= 1/2 else 2 * (1-x)
    # logi_error = [np.abs(logi[i+1] - 4*logi[i]*(1-logi[i])) for i in range(n - 1)]
    # tent_error = [np.abs(tent[i + 1] - t(tent[i])) for i in range(n - 1)]
    #
    # # print(tent.shape, logi.shape)
    # # print(logi)
    #
    # plt.figure()
    # # plt.plot(np.abs(logi - np.array([homeo1(x) for x in tent])))
    # plt.plot(logi_error, label="logistic error")
    # plt.plot(tent_error, label="tent error")
    # plt.legend()
    # plt.scatter(tent, logi)

    tent2d = dy.embedding(tent, 2, 0)
    logi2d = dy.embedding(logi, 2, 0)
    # logi3992d = dy.embedding(logi399, 2, 0)

    tent = tent.reshape((n, 1))
    logi = logi.reshape((n, 1))
    # logi399 = logi399.reshape((n, 1))

    # print(tent2d)
    # print(logi2d)
    plt.figure()
    plt.scatter(tent2d[:,0], tent2d[:,1])
    plt.scatter(logi2d[:, 0], logi2d[:, 1])

    to_compare = [tent, logi, tent2d, logi2d]
    labels = ["tent", "logistic", "tent2", "logistic2"]

    # ### FNN for tent and logi
    # r_values = range(1,12)
    # for (i, j) in combinations(range(len(to_compare)), 2):
    #     print(labels[i] + " vs. " + labels[j])
    #     ts1 = to_compare[i]
    #     ts2 = to_compare[j]
    #     new_n = min(len(ts1), len(ts2))
    #     print(ts1.shape, ts2.shape)
    #     fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=r_values)
    #     plt.figure()
    #     plt.plot(fnn1, label=labels[i] + " to " + labels[j])
    #     plt.plot(fnn2, label=labels[j] + " to " + labels[i])
    #     plt.legend()
    plt.show()

    ### KNN for tent and logi
    k_values = [1, 4, 8, 12]
    knns = np.zeros((len(to_compare), len(to_compare), len(k_values)))
    for (i, j) in combinations(range(len(to_compare)), 2):
        print(labels[i] + " vs. " + labels[j])
        ts1 = to_compare[i]
        ts2 = to_compare[j]
        new_n = min(len(ts1), len(ts2))
        knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=k_values)
        knns[i, j, :] = knn1
        knns[j, i, :] = knn2

    for ki, kv in enumerate(k_values):
        knn_df = pd.DataFrame(data=knns[:,:,ki],  index=labels, columns=labels)
        # knn_df.name
        print("K = " + str(kv))
        print(knn_df)
        print(" ")
        np.savetxt('tent_logi_knns_k'+str(kv)+'.csv', knns[:,:,ki], delimiter=",")

    # for k in range(1, 10):
    #     # print(dy.conjugacy_knn(tent.reshape((n,1)), logi.reshape((n,1)), k=k))
    #     print(dy.conjugacy_test_knn(tent2d, logi2d, k=k))
    #
    # fnns1 = []
    # fnns2 = []
    # f1,f2 = dy.fnn(tent.reshape((n, 1)), logi.reshape((n, 1)), r=range(1, 12))
    # fnns1.append(f1)
    # fnns2.append(f2)
    # f1, f2 = dy.fnn(dy.embedding(tent, 2, 0), dy.embedding(logi, 2, 0), r=range(1, 12))
    # fnns1.append(f1)
    # fnns2.append(f2)
    #
    # # fnns.append([dy.fnn(tent.reshape((n,1)), logi.reshape((n,1)), r=r)[0] for r in range(1,12)])
    # # fnns.append([dy.fnn(dy.embedding(tent, 2, 0), dy.embedding(logi, 2, 0), r=r)[0] for r in range(1, 12)])
    #
    # plt.figure()
    # i = 1
    # for fnn in fnns1:
    #     plt.plot(range(1,len(fnn)+1), fnn, label=str(i))
    #     i += 1
    # plt.legend()
    # plt.figure()
    # i = 1
    # for fnn in fnns2:
    #     plt.plot(range(1,len(fnn)+1), fnn, label=str(i))
    #     i += 1
    # plt.legend()
    # plt.show()

def circle_rotation_test():
    n = 1000
    rot1 = 0.1
    rot2 = -0.1
    points1sx = dy.circle_rotation(n, step=rot1, starting_point=np.array([1, 0]))
    points1sy = dy.circle_rotation(n, step=rot1, starting_point=np.array([0, 1]))
    points2sx = dy.circle_rotation(n, step=rot2, starting_point=np.array([1, 0]))
    points2sy = dy.circle_rotation(n, step=rot2, starting_point=np.array([0, 1]))

    # plt.figure()
    # plt.scatter(points1sy[:,0], points1sy[:,1], s=0.1)
    # plt.show()

    # to_compare = [points1sx, points1sy, points2sx, points2sy]
    # labels = ["rot 1, st 1", "rot 1, st 2", "rot 2, st 1", "rot 2, st 2"]

    to_compare = [points1sx, points2sx, points2sy]
    labels = ["rot 1, st 1", "rot 2, st 1", "rot 2, st 2"]

    plt.figure()
    plt.plot(points2sy[:120,0])

    ### FNN for rotation
    r_values = range(1,12)

    pairs = list(combinations(range(len(to_compare)), 2))
    width_plt = int(np.ceil(np.sqrt(len(pairs))))
    print(len(pairs), width_plt * (width_plt - 1))
    height_plt = width_plt - 1 if len(pairs) <= width_plt * (width_plt - 1) else width_plt

    fig = plt.figure()
    print(width_plt, height_plt)
    for pidx, (i, j) in enumerate(pairs):
        print(labels[i] + " vs. " + labels[j], pidx)
        ts1 = to_compare[i]
        ts2 = to_compare[j]
        new_n = min(len(ts1), len(ts2))
        print(ts1.shape, ts2.shape)
        fnn1, fnn2 = dy.fnn(ts1[:new_n], ts2[:new_n], r=r_values)
        # plt.figure()
        ax = fig.add_subplot(width_plt, height_plt, pidx+1)
        ax.plot(fnn1, label=labels[i] + " to " + labels[j])
        ax.plot(fnn2, label=labels[j] + " to " + labels[i])
        ax.legend()

    ### KNN for tent and logi
    k_values = [1, 4, 8, 12]
    knns = np.zeros((len(to_compare), len(to_compare), len(k_values)))
    for (i, j) in combinations(range(len(to_compare)), 2):
        print(labels[i] + " vs. " + labels[j])
        ts1 = to_compare[i]
        ts2 = to_compare[j]
        new_n = min(len(ts1), len(ts2))
        knn1, knn2 = dy.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=k_values)
        knns[i, j, :] = knn1
        knns[j, i, :] = knn2

    for ki, kv in enumerate(k_values):
        knn_df = pd.DataFrame(data=knns[:,:,ki],  index=labels, columns=labels)
        # knn_df.name
        print("K = " + str(kv))
        print(knn_df)
        print(" ")
        np.savetxt('rotation_knns_k'+str(kv)+'.csv', knns[:,:,ki], delimiter=",")

    plt.show()

def conjugacy_nonaligned_series():
    # n = 1000
    # rot1 = 0.1
    # rot2 = 0.11
    # points1 = dy.circle_rotation(n, step=rot1, starting_point=np.array([1, 0]))
    # points2 = dy.circle_rotation(n, step=rot2, starting_point=np.array([1, 0]))
    #
    # def h(x):
    #     return x
    #
    # print(dy.conjugacy_test(points1, points2, h, k=5))

    import sympy
    n = 2000
    sp = 0.2

    def homeo(x):
        return 2 * np.arcsin(np.sqrt(x)) / pi
    def homeo1(x):
        return np.sin(pi * x / 2)**2
    def id(x):
        return x
    def t(x):
        return 2 * x if x<= 1/2 else 2 * (1-x)

    logi = np.array(dy.logistic_map(n, starting_point=sp, r=3.9))
    logi2 = np.array(dy.logistic_map(n, starting_point=sp+0.15, r=3.9))
    # tent = np.array(dy.logistic_map(n, r=4.00, starting_point=sp))
    tent = np.array([homeo(x) for x in logi2], dtype=float)
    # tent = np.array(dy.logistic_map(n, starting_point=sp+0.2, r=3.9))

    # plt.figure()
    # plt.plot(logi)
    # plt.plot(tent)
    # plt.show()
    # print(logi[:-10])
    # print(tent[:-10])

    tent = tent.reshape((n, 1))
    logi = logi.reshape((n, 1))


    # print(dy.conjugacy_test(logi, tent, homeo))

    logi_400_1 = np.array(dy.logistic_map(n, starting_point=sp, r=4.0))
    logi_395_1 = np.array(dy.logistic_map(n, starting_point=sp, r=3.95))
    # logi_390_1 = np.array(dy.logistic_map(n, starting_point=sp, r=3.9))
    logi_395_2 = np.array(dy.logistic_map(n, starting_point=sp+0.2, r=3.95))

    tent_400_1 = np.array([homeo(x) for x in logi_400_1], dtype=float)
    tent_395_1 = np.array([homeo(x) for x in logi_395_1], dtype=float)
    # tent_390_1 = np.array([homeo(x) for x in logi_390_1], dtype=float)
    tent_395_2 = np.array([homeo(x) for x in logi_395_2], dtype=float)

    to_compare = [logi_400_1, logi_395_1, logi_395_2, tent_400_1, tent_395_1, tent_395_2]
    labels = ["logi_400_1", "logi_395_1", "logi_395_2", "tent_400_1", "tent_395_1", "tent_395_2"]


    diffs = np.zeros((len(to_compare), len(to_compare)))
    for (i, j) in combinations(range(len(to_compare)), 2):
        print((i,j), " " + labels[i] + " vs. " + labels[j])
        ts1 = to_compare[i].reshape((n, 1))
        ts2 = to_compare[j].reshape((n, 1))
        new_n = min(len(ts1), len(ts2))
        if i <= 2 and j > 2:
            hf = homeo
            hb = homeo1
        else:
            hf = id
            hb = id

        diffs[i, j] = dy.conjugacy_test(ts1[:new_n], ts2[:new_n], hf, k=2)
        diffs[j, i] = dy.conjugacy_test(ts2[:new_n], ts1[:new_n], hb, k=2)


    df = pd.DataFrame(data=diffs,  index=labels, columns=labels)
    # display(df.to_string())
    print(df.to_markdown())
    # print(knn_df)
    print(" ")
    # np.savetxt('rotation_knns_k'+str(kv)+'.csv', knns[:,:,ki], delimiter=",")

if __name__ == '__main__':
    # short_test()
    # lorenz_test()
    # fnn_test()
    # logi_tent_test()
    # circle_rotation_test()
    conjugacy_nonaligned_series()
