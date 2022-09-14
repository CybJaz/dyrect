# from collections import Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import re
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist


def trans2prob(matrix):
    # each row sums to 1 and represenst outward probabilities of a transition
    row_sums = np.sum(matrix, axis=1)
    return matrix / row_sums[:, None]


def embedding(points, dimension, delay):
    emb_indices = np.arange(dimension) * (delay + 1) + np.arange(
        np.max(points.shape[0] - (dimension - 1) * (delay + 1), 0)).reshape(-1, 1)
    ep = points[emb_indices]
    if len(points.shape) == 1:
        return ep
    else:
        return ep[:, :, 0]


class TransitionMatrix:
    def __init__(self, landmarks, epsilon, p=2, alpha=True, prob=True):
        """
        :param landmarks:
        :param epsilon:
        :param p:
        :param alpha: boolean, do cells should be counted as alpha shapes?
        :param prob: is the resulting matrix a probability matrix (True) or a binary matrix (False)
        """
        self._landmarks = landmarks
        self._nlands = len(landmarks)
        # p defines norm type
        self._p = p
        self._eps = epsilon
        self._alpha = alpha
        self._prob = prob

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        # compute distances
        dist_to_landmarks = distance_matrix(X, self._landmarks, p=self._p)
        trans_mat = np.full((self._nlands, self._nlands), 0.)

        if self._alpha:
            prev_min = np.min(dist_to_landmarks[0])
            assert prev_min <= self._eps
            prev = np.where(dist_to_landmarks[0] == prev_min)[0]
            for i in range(1, len(X)):
                curr_min = np.min(dist_to_landmarks[i])
                assert curr_min <= self._eps
                curr = np.where(dist_to_landmarks[i] == curr_min)[0]
                fr, to = np.meshgrid(prev, curr)
                trans_mat[fr, to] += 1
                prev = curr
        else:
            # TODO: check if this way of counting transitions is reasonable
            prev = np.where(dist_to_landmarks[0] < self._eps)[0]
            for i in range(1, len(X)):
                curr = np.where(dist_to_landmarks[i] < self._eps)[0]
                fr, to = np.meshgrid(prev, curr)
                trans_mat[fr, to] += 1
                prev = curr

        if self._prob:
            return trans_mat
        else:
            return np.where(trans_mat>0, 1, 0)


class GeomTransitionMatrix:
    def __init__(self, landmarks, scomplex, epsilon, p=2, alpha=True):
        """
        :param landmarks:
        :param epsilon:
        :param p:
        :param alpha: boolean, do cells should be counted as alpha shapes, aka mitosis?
        """
        self._landmarks = landmarks
        self._nlands = len(landmarks)
        # p defines norm type
        self._p = p
        self._eps = epsilon
        self._complex = scomplex
        self._alpha = alpha

    def fit(self, X, Y=None):
        """
        Fit a sequenc
        @param X: a sequence of trajectories
        @param Y: nothing
        @return:
        """

        trans_mat = np.full((self._nlands, self._nlands), 0.)

        # fit for every trajectory
        for sX in X:
            # TODO: what if alpha is false
            # compute distances
            dist_to_landmarks = distance_matrix(sX, self._landmarks, p=self._p)
            closest_landmark = np.argmin(dist_to_landmarks, axis=1)

            prev = closest_landmark[0]
            assert dist_to_landmarks[0, prev] <= self._eps

            # not interpolable skips
            non_lin_skips = []

            for i in range(1, len(sX)):
                curr = closest_landmark[i]
                assert dist_to_landmarks[i, curr] <= self._eps
                # increment transition if we moved from one cover element to the other (or we stayed in the same)
                # otherwise try to interpolate midpoints
                if prev == curr or tuple(np.sort([prev, curr])) in self._complex.simplices[1]:
                    trans_mat[prev, curr] += 1
                else:
                    mid_points = np.linspace(sX[i - 1], sX[i], 7)
                    mid_dists = distance_matrix(mid_points[1:-1], self._landmarks, p=self._p)
                    mid_lms = [np.argmin(mdists) for mdists in mid_dists]
                    # interpolated path
                    ip = np.concatenate(([prev], mid_lms, [curr]))
                    for l in range(len(ip) - 1):
                        if ip[l] != ip[l + 1]:
                            if tuple(np.sort([ip[l], ip[l + 1]])) in self._complex.simplices[1]:
                                trans_mat[ip[l], ip[l + 1]] += 1
                            else:
                                non_lin_skips.append(i)
                prev = curr

            # fix points that can't be linearly interpolated by searching for the shortest path in the graph
            if len(non_lin_skips) > 0:
                dg = nx.from_numpy_array(np.where(trans_mat > 0, 1, 0), create_using=nx.DiGraph)

                for i in range(len(non_lin_skips)):
                    x0 = non_lin_skips[i]
                    src = closest_landmark[x0 - 1]
                    trg = closest_landmark[x0]
                    try:
                        paths = [p for p in nx.all_shortest_paths(dg, source=src, target=trg)]
                        weight = 1. / len(paths)
                        #                     print([p for p in paths], [src, trg])
                        for path in paths:
                            for l in range(len(path) - 1):
                                trans_mat[path[l], path[l + 1]] += weight

                    # catch nx.NetworkXNoPath:
                    # print("no path from " + src + " to " + trg)
                    except nx.NetworkXNoPath as err:
                        print("no path from " + str(src) + " to " + str(trg))
                    else:
                        print("something different")


        return trans_mat


def symbolization(X, lms, eps=0, simplified=False):
    """ symbolization of a time series, a sequence of points is transformed into a list of integers (elements of
            the lms cover)
        X - a time series of a shape [n, dim]
        lms - landmarks - an array of shape [l, dim]
        eps - epsilon (from epsilon net)
        TODO: simplified - reduce repeating consecutive symbols
    """
    distances = cdist(X, lms, 'euclidean')  # distances to landmarks
    symbols = np.array([np.argmin(point_to_lms) for point_to_lms in distances])

    # check if points in X are covered
    if eps > 0:
        symbols = np.array([(l if distances[i, l] < eps else -1) for i, l in enumerate(symbols)])
    return symbols


def symb2string(symbols, codesize=-1):
    """
    Transforms a list (symbols) of positive integers into a string
    :param symbols: a list of integers
    :param codesize:
    :return:
    """
    m = np.max(symbols)
    if codesize < 1:
        codesize = len(str(m))
    nums = [''.join([str(0) for _ in range(codesize - len(str(x)))]) + str(x) for x in symbols]
    return '-'.join(nums)


@dataclass
class Future:
    sequence: tuple
    counter: int
    occurences: list


@dataclass
class Prediction:
    past: tuple
    futures: list


# history - a training time series
# history book - a symbolized training time series
# past/query - a new short time series
# story - a symbolized query
# future - a

class Seer:
    """
    A class for making predictions from a time series based on an epsilon-net
    """

    def __init__(self, history, cover, eps=-1, simplified=False):
        """
        :param history: a time series used to create the database for predictions
        :param cover: a set of landmarks defining the set of symbols
        :param eps: if eps is -1 we are looking just for the closest landmark otherwise a distance less then epsilon is
            required, if distance is larger than eps we put extra symbol '-1'
        :param TODO: simplified - reduce repeating consecutive symbols
        """
        # cover variable could be more abstract, for now it's just a collection of landmarks
        self._history = history
        self._cover = cover
        self._eps = eps
        codes = symbolization(history, cover, eps)
        self._codesize = len(str(max(codes)))
        self._history_book = symb2string(codes)
        self._dimension = len(history[0])
        assert len(cover[0]) == self._dimension

        # it's state-machine like variables
        self._recent_reg = None
        self._recent_query = None
        self._recent_story = None
        self._recent_futures = None

        self._recent_prediction = None

        # TODO:
        # class Query:

    def predict(self, past, f, p=-1):
        dim = len(past[0])
        assert len(past[0]) == self._dimension

        # (p,f) - p-past steps, f-future steps predictions
        # TODO: p is not used
        if p == -1:
            p = len(past)
        else:
            p = min(len(past), p)

        self._recent_query = past[-p:]
        self._recent_story = symbolization(self._recent_query, self._cover, self._eps)
        if min(self._recent_story) < 0:
            # the negative symbol means an unknown symbol
            print("this past has never happened before")
            return None
        self._recent_reg = symb2string(self._recent_story, codesize=self._codesize) + '.{' + str(f * (self._codesize+1)) + '}'
        # print(reg)
        futures = [(event.group(0), event.span(0)) for event in re.finditer(self._recent_reg, self._history_book)]
        # print(futures[0])
        futures = [(tuple([int(k) for k in f[1:].split('-')]), idxs) for (f, idxs) in futures]
        # self._recent_unique_futures = Counter([future[0][-f * (self._codesize+1):] for future in futures])
        futures_dict = dict()
        for f, idxs in futures:
            (b,e) = (int(idxs[0] / (self._codesize+1.)), int((idxs[1] + 1) / (self._codesize+1.)))
            if f in futures_dict:
                futures_dict[f].counter += 1
                futures_dict[f].occurences.append((b,e))
            else:
                futures_dict[f] = Future(f, 1, [(b,e)])

        self._recent_futures = [f for f in futures_dict.values()]
        self._recent_futures.sort(key=(lambda f: f.counter), reverse=True)

        self._recent_prediction = Prediction(self._recent_story, self._recent_futures)
        return self._recent_prediction

    def draw_hom_grouped_prediction(self, complex, steps=[], prediction=None):
        """
        :param prediction: a list of Future objects, if None draw the recent one
        :return:
        """
        if prediction is None:
            if self._recent_prediction is None:
                print("no prediction to draw")
                return None
            else:
                prediction = self._recent_prediction

        def the_same(path1, path2):
            if len(path1) == len(path2) and all([path1[i] == path2[i] for i in range(len(path1))]):
                return True
            return False

        paths = np.array([list(f.sequence) for f in prediction.futures])

        t0 = len(prediction.past)
        t1 = len(paths[0])
        period = t1-t0

        t_subcomplexes = []
        t_components = []
        for t in range(t0, t1):
            tsnc, components = complex.subcomplex(paths[:, t])
            t_subcomplexes.append(tsnc)
            t_components.append(
                [set([i for i in range(len(paths)) if paths[i, t] in component]) for component in components])

        t_clusters = {0: t_components[0]}
        for t in range(1, period):
            t_clusters[t] = []
            for c1 in t_clusters[t - 1]:
                for c2 in t_components[t]:
                    c3 = c1.intersection(c2)
                    if len(c3) > 0:
                        t_clusters[t].append(c3)

        lc = 0
        for c in t_clusters:
            nlc = len(t_clusters[c])
            if nlc != lc:
                print("T: ", c, " components: ", nlc)
                lc = nlc

        for step in steps:
            print("T: ", step)
            clusters_at_step = t_clusters[step]

            past_paths_in_clusters = []
            for idx, cluster in enumerate(clusters_at_step):
                paths_in_cluster = []
                for el in cluster:
                    for (b, e) in prediction.futures[el].occurences:
                        paths_in_cluster.append(self._history[b:(b + t0 + step)])
                past_paths_in_clusters.append(paths_in_cluster)
            # past_paths_in_clusters

            fig = plt.figure(figsize=(12, 10), dpi=80)

            colors = plt.cm.rainbow(np.linspace(0, 1, len(past_paths_in_clusters)))

            d = len(past_paths_in_clusters[0][0][0])
            # print(d, past_paths_in_clusters[0])

            if d == 3:
                ax = fig.add_subplot(projection='3d')
                for idx, cluster_paths in enumerate(past_paths_in_clusters):
                    for path in cluster_paths:
                        ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=0.5, color=colors[idx])
            elif d == 2:
                ax = fig.add_subplot()
                for idx, cluster_paths in enumerate(past_paths_in_clusters):
                    for path in cluster_paths:
                        ax.plot(path[:, 0], path[:, 1], linewidth=0.5, color=colors[idx])

        plt.show()
        return fig, ax

    def draw_prediction(self, prediction=None):
        """
        :param prediction: a list of Future objects, if None draw the recent one
        :return:
        """
        if prediction is None:
            if self._recent_prediction is None:
                print("no prediction to draw")
                return None
            else:
                prediction = self._recent_prediction

        # TODO: how to pass fig arguments as an argument
        fig = plt.figure(figsize=(12, 10), dpi=80)

        if self._dimension == 3:
            ax = fig.add_subplot(projection='3d')

            # a combinatorial trajectory
            combinatorial_past = np.array([self._cover[k] for k in prediction.past])

            ax.scatter(combinatorial_past[:, 0], combinatorial_past[:, 1], combinatorial_past[:, 2], c='black', s=30)
            ax.plot(combinatorial_past[:, 0], combinatorial_past[:, 1], combinatorial_past[:, 2], c='black', linewidth=4)

            for cpath in prediction.futures:
                path = np.array([self._cover[k] for k in cpath.sequence[(len(prediction.past)):]])
                ax.plot([combinatorial_past[-1, 0], path[0, 0]], [combinatorial_past[-1, 1], path[0, 1]], \
                        [combinatorial_past[-1, 2], path[0, 2]], linewidth=1.5)
                ax.scatter(path[:, 0], path[:, 1], path[:, 2])
                ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=2)

            for future in prediction.futures:
                for (b, e) in future.occurences:
                    ax.plot(self._history[b:e, 0], self._history[b:e, 1], self._history[b:e, 2], linewidth=0.2)

        elif self._dimension == 2:
            ax = fig.add_subplot()
            combinatorial_past = np.array([self._cover[k] for k in prediction.past])

            ax.scatter(combinatorial_past[:, 0], combinatorial_past[:, 1], c='black', s=30)
            ax.plot(combinatorial_past[:, 0], combinatorial_past[:, 1], c='black',
                    linewidth=4)

            for cpath in prediction.futures:
                path = np.array([self._cover[k] for k in cpath.sequence[(len(prediction.past)):]])
                if len(path) > 0:
                    ax.plot([combinatorial_past[-1, 0], path[0, 0]], [combinatorial_past[-1, 1], path[0, 1]],
                            linewidth=1.5)
                    ax.scatter(path[:, 0], path[:, 1])
                    ax.plot(path[:, 0], path[:, 1], linewidth=2)

            for future in prediction.futures:
                for (b, e) in future.occurences:
                    ax.plot(self._history[b:e, 0], self._history[b:e, 1], linewidth=0.2)
        return fig, ax