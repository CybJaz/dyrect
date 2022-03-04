from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
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
    return points[emb_indices]


class TransitionMatrix:
    def __init__(self, landmarks, epsilon, p=2):
        self._landmarks = landmarks
        self._nlands = len(landmarks)
        # p defines norm type
        self._p = p
        self._eps = epsilon

    def fit(self, X, Y=None):
        # compute distances
        dist_to_landmarks = distance_matrix(X, self._landmarks, p=self._p)
        trans_mat = np.full((self._nlands, self._nlands), 0.)

        prev = np.where(dist_to_landmarks[0] < self._eps)[0]
        for i in range(1, len(X)):
            curr = np.where(dist_to_landmarks[i] < self._eps)[0]
            fr, to = np.meshgrid(prev, curr)
            trans_mat[fr, to] += 1
            prev = curr

        return trans_mat


def symbolization(X, lms, eps=0):
    """ symbolization of a time series, a sequence of points is transformed into a list of integers (elements of
            the lms cover)
        X - a time series of a shape [n, dim]
        lms - landmarks - an array of shape [l, dim]
        eps - epsilon (from epsilon net)
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


class Seer:
    """
    A class for making predictions from a time series based on an epsilon-net
    """

    def __init__(self, history, cover, eps=-1):
        # cover variable could be more abstract, for now it's just a collection of landmarks
        self._history = history
        self._cover = cover
        self._eps = eps
        codes = symbolization(history, cover, eps)
        self._codesize = len(str(max(codes)))
        self._history_book = symb2string(codes)

    def predict(self, past, f, draw=False):
        # (p,f) - p-past steps, f-future steps predictions
        p = len(past)
        story = symbolization(past, self._cover, self._eps)
        if min(story) < 0:
            print("this past has never happened before")
            return []
        reg = symb2string(story, codesize=self._codesize) + '.{' + str(f * 4) + '}'
        print(reg)
        futures = [(event.group(0), event.span(0)) for event in re.finditer(reg, self._history_book)]
        #         re.findall(reg, self.history_book_)
        unique_futures = Counter([future[0][-f * 4:] for future in futures])
        #         unique_futures = Counter([future[0] for future in futures])

        #         print([(event.group(0), event.span(0)) for event in re.finditer(reg, self.history_book_)])

        if draw:
            fig = plt.figure(figsize=(12, 10), dpi=80)

            ax = fig.add_subplot(projection='3d')

            past_path = np.array([self._cover[int(k)] for k in story])
            # print(past_path)

            ax.scatter(past_path[:, 0], past_path[:, 1], past_path[:, 2], c='black', s=30)
            ax.plot(past_path[:, 0], past_path[:, 1], past_path[:, 2], c='black', linewidth=4)

            for idx, key in enumerate(list(unique_futures.keys())):
                path = np.array([self._cover[int(k)] for k in key[1:].split('-')])
                ax.scatter(path[:, 0], path[:, 1], path[:, 2])
                ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=2)

            for (_, (b, e)) in futures:
                #                 print(b,e)
                #                 print(b/4,(e+1)/4)
                (b, e) = (int(b / 4), int((e + 1) / 4))
                #                 print(self.history_book_[b:e])
                ax.plot(self._history[b:e, 0], self._history[b:e, 1], self._history[b:e, 2], linewidth=0.2)
        #             plt.show()
        return unique_futures
