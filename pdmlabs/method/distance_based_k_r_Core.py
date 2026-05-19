from numpy._typing import NDArray
import numpy as np
from scipy.spatial.distance import cdist


class distance_based_k_r:
    def __init__(self, k=5, window_norm=False, metric="euclidean"):
        self.k = k
        self.window_norm = window_norm
        self.metric = metric
        self.to_fit = None

    def fit(self, df: NDArray):
        self.to_fit = df
        if len(self.to_fit)<=self.k:
            self.k=len(self.to_fit)-1
    def predict(self, df: NDArray):
        to_predict = df
        D, _ = self._search(to_predict,self.to_fit)
        score = []
        for d in D[:, self.k - 1]:
            score.append(d)
        return score

    def _calc_dist(self, query: NDArray, pts: NDArray):
        return cdist(query, pts, metric=self.metric)

    def _search(self, query: NDArray, points: NDArray):
        dists = self._calc_dist(query, points)

        I = (
            np.argsort(dists, axis=1)
            if self.k > 1
            else np.expand_dims(np.argmin(dists, axis=1), axis=1)
        )
        D = np.take_along_axis(np.array(dists), I, axis=1)
        return D, I
