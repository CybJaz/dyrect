import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff

def conjugacy_test_knn(ts1, ts2, k = [1], point_vals=False):
    """
    The assumption the conjugacy h:X->Y maps h(x_i) = y_i
    :param ts1: time series one - an array of size [n, d]
    :param ts2: time series two - shape the same as ts1
    :return:
    """
    assert len(ts1) == len(ts2)
    n = len(ts1)

    dists1 = cdist(ts1, ts1)
    dists2 = cdist(ts2, ts2)
    nn1 = np.argsort(dists1, axis=1)
    nn2 = np.argsort(dists2, axis=1)

    knns_first_vs_second = []
    local_diffs_1 = []
    knns_second_vs_first = []
    local_diffs_2 = []
    for kv in k:
        diff12 = 0
        pvals12 = []
        pvals21 = []
        # compare ts1 to ts2
        for i in range(n):
            knn1 = set(nn1[i,:kv+1])
            diff = -len(knn1)
            j = 0
            # print(nn1[i,:10], nn2[i,:10])
            while len(knn1) > 0:
                knn1 = knn1.difference([nn2[i, j]])
                diff += 1
                j += 1
            pvals12.append(diff)
            diff12 += diff
        diff12 = diff12 / (n * n)
        knns_first_vs_second.append(diff12)
        local_diffs_1.append(pvals12)

    for kv in k:
        diff21 = 0
        # compare ts2 to ts1
        for i in range(n):
            knn2 = set(nn2[i,:kv+1])
            diff = -len(knn2)
            j = 0
            while len(knn2) > 0:
                knn2 = knn2.difference([nn1[i,j]])
                diff += 1
                j += 1
            diff21 += diff
            pvals21.append(diff)
        diff21 = diff21 / (n * n)
        knns_second_vs_first.append(diff21)
        local_diffs_2.append(pvals21)

    if point_vals:
        return knns_first_vs_second, knns_second_vs_first, local_diffs_1, local_diffs_2
    else:
        return knns_first_vs_second, knns_second_vs_first

def symmetric_conjugacy_knn(ts1, ts2, k):
    """
    The assumption the conjugacy h:X->Y maps h(x_i) = y_i
    :param ts1: time series one - an array of size [n, d]
    :param ts2: time series two - shape the same as ts1
    :return:
    """
    assert len(ts1) == len(ts2)
    n = len(ts1)

    dists1 = cdist(ts1, ts1)
    dists2 = cdist(ts2, ts2)
    nn1 = np.argsort(dists1, axis=1)
    nn2 = np.argsort(dists2, axis=1)

    total_diff = 0
    for i in range(n):
        knn1 = set(nn1[i,:k+1])
        diff1 = -len(knn1)
        j = 0
        # print(nn1[i,:10], nn2[i,:10])
        while len(knn1) > 0:
            knn1 = knn1.difference([nn2[i, j]])
            diff1 += 1
            j += 1

        knn2 = set(nn2[i,:k+1])
        diff2 = -len(knn2)
        j = 0
        while len(knn2) > 0:
            knn2 = knn2.difference([nn1[i,j]])
            diff2 += 1
            j += 1
        total_diff = max(diff1, diff2)

    return total_diff / n

def fnn(ts1, ts2, r=[1]):
    assert len(ts1) == len(ts2)
    n = len(ts1)
    dists1 = []
    dists2 = []
    for i in range(n):
        for j in range(i+1, n):
            dists1.append(np.linalg.norm(ts1[i] - ts1[j]))
            dists2.append(np.linalg.norm(ts2[i] - ts2[j]))
    std1 = np.std(dists1)
    std2 = np.std(dists2)

    dists1 = cdist(ts1, ts1)
    dists2 = cdist(ts2, ts2)
    dists1 = dists1 + np.diag(np.ones((n,)) * 2 * np.max(dists1))
    dists2 = dists2 + np.diag(np.ones((n,)) * 2 * np.max(dists2))

    # nearest neighbors
    nn1 = np.argmin(dists1, axis=1)
    nn2 = np.argmin(dists2, axis=1)

    def H(x):
        return 1 if x > 0 else 0

    fnns1 = []
    fnns2 = []
    for rv in r:
        fnn1_div = 0
        fnn2_div = 0
        fnn1_num = 0
        fnn2_num = 0
        for i in range(n):
            v1 = H(std1/rv - dists1[i, nn1[i]])
            fnn1_num += H(dists2[i, nn1[i]]/dists1[i, nn1[i]] - rv) * v1
            fnn1_div += v1

            v2 = H(std2 / rv - dists2[i, nn2[i]])
            fnn2_num += H(dists1[i, nn2[i]] / dists2[i, nn2[i]] - rv) * v2
            fnn2_div += v2
        fnns1.append(fnn1_num / fnn1_div)
        fnns2.append(fnn2_num / fnn2_div)

    return fnns1, fnns2

def conjugacy_test(tsX, tsY, h, k = 1):
    """
    Conjugacy that does not require the direct correspondence of time series
    :param tsX: a time series in X
    :param tsY: a time series in Y
    :param h: a map from X to Y
    :return:
    """

    distsX = cdist(tsX[:-1], tsX[:-1])
    # distsY = cdist(tsX, tsY)

    nnX = np.argsort(distsX, axis=1)
    # nnY = np.argsort(distsY, axis=1)
    # print(tsX.shape, tsY.shape)

    accumulated_hausdorff = []
    for i in range(len(tsX)-1):
        idx_knnX = nnX[i, :k+1]
        hknnX = h(np.array([tsX[x] for x in idx_knnX]))
        # print(hknnX)
        # print(hknnX.shape)
        if len(hknnX) == 1:
            hknnX = hknnX.reshape((1, 1))
        knns_dists = cdist(hknnX, tsY[:-1])
        idx_knnY = np.argmin(knns_dists, axis=1)
        hfknnX = h(np.array([tsX[x+1] for x in idx_knnX]))
        gknnY =  np.array([tsY[y+1] for y in idx_knnY])

        # print(np.array([tsX[x] for x in idx_knnX]).shape, np.array([tsY[y] for y in idx_knnY]).shape)
        dom_hdist = directed_hausdorff(np.array([tsX[x] for x in idx_knnX]), np.array([tsY[y] for y in idx_knnY]))
        im_hdist = directed_hausdorff(hfknnX, gknnY)

        # hdist2 = directed_hausdorff(gknnY, hfknnX)
        # print(dom_hdist, im_hdist)
        if dom_hdist[0] == 0:
            accumulated_hausdorff.append(im_hdist[0]/0.00001)
        else:
            accumulated_hausdorff.append(im_hdist[0] / dom_hdist[0])

    return np.sum(accumulated_hausdorff) / (len(tsX) - 1)