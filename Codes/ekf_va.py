"""EKF that replaces NLOS-flagged real-anchor RSSI with VA pseudo-RSSI measurement.

Key stability fix vs. naive VA design: the VA pseudo-measurement is generated from the
filter's OWN estimate, which creates a tight self-referential feedback loop (a biased
estimate gets "confirmed" by its own VA and the bias compounds). We break this loop by:
  1) using an EMA-smoothed position (lagged, lower-variance) instead of the instantaneous
     just-predicted state to generate the VA pseudo-RSSI and to drive GDOP optimization;
  2) trusting the VA measurement less than a real anchor (SIGMA_VA > SIGMA_LOS);
  3) enforcing a minimum VA standoff distance (avoids near-field log-distance blow-up);
  4) gating the update so a single bad noise draw can't inject a large state jump.
"""
import numpy as np
from ekf import EKF, DT
from ekf_nees import EKF_NEES
from sensors import SIGMA_LOS
from va_optimizer import optimize_va
from va_rssi import generate_va_rssi, SIGMA_VA

EMA_ALPHA = 0.15  # smoothing factor for the position estimate used to drive VA logic
GATE_THRESH = 6.0  # per-row chi2 gate (~98.5% 1-DOF) for soft-reject in update()
GLOBAL_NIS_GATE = 14.0  # chi2(8 dof, ~92%); skip update entirely if whole-vector NIS exceeds this


class EKF_VA(EKF_NEES):
    def __init__(self, x0, P0=None, n_anchors=6):
        super().__init__(x0, P0, n_anchors)
        self.va_cache = {}
        self.va_positions = {i: None for i in range(n_anchors)}
        self.smoothed_pos = np.array(x0[:2], dtype=float)

    def step(self, z_rssi_raw, acc, anchors, rng, dt=DT):
        self.predict(dt)

        sigma_nom = np.full(self.n_anchors, SIGMA_LOS)
        H, R = self.build_H_R(anchors, None, sigma_nom)
        z_hat = self.predicted_z(anchors)
        y_full = np.concatenate([z_rssi_raw, acc]) - z_hat
        S = H @ self.P @ H.T + R
        S_diag_rssi = np.diag(S)[: self.n_anchors]
        y_rssi = y_full[: self.n_anchors]

        eps = self.detector.per_anchor_nees(y_rssi, S_diag_rssi)
        flags = self.detector.update(eps)

        eff_anchor_pos = anchors.copy().astype(float)
        z_eff = z_rssi_raw.copy()
        sigma_eff = np.full(self.n_anchors, SIGMA_LOS)

        # EMA-smoothed position: decouples VA generation from the instantaneous (about-to-be-
        # corrected-by-this-very-measurement) filter state, breaking the feedback loop.
        tag_smoothed = self.smoothed_pos
        los_anchor_positions = [anchors[i] for i in range(self.n_anchors) if not flags[i]]
        if len(los_anchor_positions) < 2:
            los_anchor_positions = list(anchors)

        for i in range(self.n_anchors):
            if flags[i]:
                va_pos, _ = optimize_va(los_anchor_positions, tag_smoothed, cache=self.va_cache, cache_key=i)
                self.va_positions[i] = va_pos
                eff_anchor_pos[i] = va_pos
                z_eff[i] = generate_va_rssi(va_pos, tag_smoothed, rng)
                sigma_eff[i] = SIGMA_VA  # weaker trust than a real LOS anchor
            else:
                self.va_positions[i] = None

        z = np.concatenate([z_eff, acc])

        # Global safety net: if the WHOLE innovation vector is a wild outlier (e.g. a rare
        # compounding bad noise draw on top of an already-imperfect VA), skip the update this
        # step rather than let it inject a large jump -- fall back to predict-only.
        H_chk, R_chk = self.build_H_R(eff_anchor_pos, None, sigma_eff)
        y_chk = z - self.predicted_z(eff_anchor_pos)
        S_chk = H_chk @ self.P @ H_chk.T + R_chk
        try:
            nis_total = float(y_chk @ np.linalg.solve(S_chk, y_chk))
        except np.linalg.LinAlgError:
            nis_total = np.inf

        if nis_total <= GLOBAL_NIS_GATE:
            self.update(z, eff_anchor_pos, sigma_eff, gate_thresh=GATE_THRESH)

        self.smoothed_pos = (1 - EMA_ALPHA) * self.smoothed_pos + EMA_ALPHA * self.x[:2]
        return flags, eps, dict(self.va_positions)
