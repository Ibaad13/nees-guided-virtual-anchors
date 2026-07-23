"""Training-free k-NN NLOS detector: per-anchor rolling window of RSSI residuals
(actual RSSI minus log-distance-model prediction from current position estimate).
Flags NLOS via a local-outlier-factor-style k-NN distance ratio: how anomalous is
the new residual relative to the local density of the last 20 residuals for that
anchor, versus how anomalous those window points are relative to each other."""
import numpy as np

WINDOW = 20
K = 5
LOF_RATIO_THRESH = 2.0  # flag if new point's kNN distance > this x the window's own median kNN distance


class KnnNlosDetector:
    def __init__(self, n_anchors=6):
        self.n = n_anchors
        self.windows = [[] for _ in range(n_anchors)]

    def _knn_dist(self, point, others, k):
        if len(others) == 0:
            return 0.0
        d = np.abs(np.array(others) - point)
        d.sort()
        kk = min(k, len(d))
        return d[:kk].mean()

    def update(self, residuals):
        """residuals: (n_anchors,) actual RSSI - model-predicted RSSI at current pos estimate.
        Returns flags (n_anchors,) bool."""
        flags = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            w = self.windows[i]
            r = residuals[i]
            if len(w) >= K + 2:
                new_knn = self._knn_dist(r, w, K)
                # local density reference: median kNN distance among window points themselves
                local_knns = []
                for j in range(len(w)):
                    others = w[:j] + w[j + 1:]
                    local_knns.append(self._knn_dist(w[j], others, K))
                ref = np.median(local_knns) if local_knns else 1e-6
                ratio = new_knn / max(ref, 1e-6)
                flags[i] = ratio > LOF_RATIO_THRESH
            w.append(r)
            if len(w) > WINDOW:
                w.pop(0)
        return flags
