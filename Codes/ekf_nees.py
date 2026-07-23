"""EKF + per-anchor NEES detection. R_i -> 1e6 (reject) if flagged NLOS."""
import numpy as np
from ekf import EKF, DT
from nees_detector import NeesDetector
from sensors import SIGMA_LOS

R_REJECT = 1e6


class EKF_NEES(EKF):
    def __init__(self, x0, P0=None, n_anchors=6):
        super().__init__(x0, P0)
        self.detector = NeesDetector(n_anchors)
        self.n_anchors = n_anchors

    def step(self, z, anchors, dt=DT, mode="reject"):
        """mode: 'reject' -> set R_i=1e6 for flagged anchors before update.
        Returns: nlos_flags, eps (per-anchor NEES from PRE-update residual using current sigma)."""
        self.predict(dt)

        # First pass: use nominal LOS sigma to compute innovation/NEES for detection
        sigma_nom = np.full(self.n_anchors, SIGMA_LOS)
        H, R = self.build_H_R(anchors, None, sigma_nom)
        z_hat = self.predicted_z(anchors)
        y = z - z_hat
        S = H @ self.P @ H.T + R
        S_diag_rssi = np.diag(S)[: self.n_anchors]
        y_rssi = y[: self.n_anchors]

        eps = self.detector.per_anchor_nees(y_rssi, S_diag_rssi)
        flags = self.detector.update(eps)

        sigma_eff = sigma_nom.copy()
        if mode == "reject":
            sigma_eff[flags] = np.sqrt(R_REJECT)
        # mode == "naive": sigma_eff stays nominal regardless of flags (detection logged only)

        self.update(z, anchors, sigma_eff)
        return flags, eps
