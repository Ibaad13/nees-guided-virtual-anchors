"""Per-anchor NEES-based NLOS detector."""
import numpy as np

CHI2_THRESH_1DOF_95 = 3.841
WIN = 5
VOTE_REQ = 3


class NeesDetector:
    def __init__(self, n_anchors=6):
        self.n = n_anchors
        self.hist = [[] for _ in range(n_anchors)]  # sliding window of bool exceed-flags

    def per_anchor_nees(self, y_rssi, S_rssi_diag):
        """y_rssi: (6,) innovations for RSSI rows. S_rssi_diag: (6,) innovation variances.
        Returns eps (6,) NEES values."""
        eps = (y_rssi ** 2) / np.maximum(S_rssi_diag, 1e-6)
        return eps

    def update(self, eps):
        """Push per-step exceed booleans, return nlos_flags (6,) using 3/5 vote."""
        flags = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            exceed = eps[i] > CHI2_THRESH_1DOF_95
            self.hist[i].append(exceed)
            if len(self.hist[i]) > WIN:
                self.hist[i].pop(0)
            flags[i] = sum(self.hist[i]) >= VOTE_REQ and len(self.hist[i]) >= VOTE_REQ
        return flags
