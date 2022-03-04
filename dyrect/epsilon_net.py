import numpy as np


class EpsilonNet:
    # dist: True - return distances matrix, False - return boolean matrix
    def __init__(self, eps, max_num_of_landmarks=0, dist=True, method='weighted_furthest'):
        self._eps = eps
        self._max_num_of_landmarks = max_num_of_landmarks if max_num_of_landmarks > 0 else np.iinfo(np.int16).max
        self._return_distances = dist
        self._method = method
        self._nlandmarks = 0
        self._landmarks = []

    def fit(self, X, Y=None):
        """ X - [number_of_samples, number_of_features] """
        nsamples, nfeatures = X.shape
        self._nlandmarks = 1
        self._landmarks = np.array([X[0]])

        distance_to_landmarks = np.array([np.array(np.linalg.norm(X - self._landmarks[0], axis=1))])
        distance_to_cover = distance_to_landmarks[0]
        while self._nlandmarks < self._max_num_of_landmarks and np.max(distance_to_cover) >= self._eps:
            if self._method == 'furthest_point':
                furthest_point_idx = np.argmax(distance_to_cover)
            elif self._method == 'weighted_furthest':
                distance_to_cover = [d if d >= self._eps else 0 for d in distance_to_cover]
                weights = np.power(distance_to_cover / np.max(distance_to_cover), 2)
                weights = weights / np.sum(weights)
                furthest_point_idx = np.random.choice(range(nsamples), p=weights)

            self._landmarks = np.append(self._landmarks, [X[furthest_point_idx]], axis=0)
            distance_to_landmarks = np.append(distance_to_landmarks,
                                              [np.array(np.linalg.norm(X - self._landmarks[self._nlandmarks], axis=1))],
                                              axis=0)
            distance_to_cover = np.min(np.stack((distance_to_cover, distance_to_landmarks[-1])), axis=0)
            self._nlandmarks += 1

        return np.transpose(distance_to_landmarks)

    @property
    def landmarks(self):
        return self._landmarks