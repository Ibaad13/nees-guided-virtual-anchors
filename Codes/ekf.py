"""EKF: state x=[px,py,vx,vy,ax,ay]. Measurements: 6 RSSI + 2 IMU accel."""
import numpy as np
from sensors import PL_D0, N_PLE, D0

DT = 0.1
NX = 6


def F_matrix(dt=DT):
    F = np.eye(NX)
    F[0, 2] = dt
    F[1, 3] = dt
    F[0, 4] = 0.5 * dt ** 2
    F[1, 5] = 0.5 * dt ** 2
    F[2, 4] = dt
    F[3, 5] = dt
    return F


Q = np.diag([0.01, 0.01, 0.05, 0.05, 0.1, 0.1])


def h_rssi(x, anchor_pos):
    d = max(np.linalg.norm(x[:2] - anchor_pos), 0.1)
    return PL_D0 - 10 * N_PLE * np.log10(d / D0)


def H_rssi_row(x, anchor_pos):
    """dh/dx for one RSSI row, analytical."""
    dx = x[0] - anchor_pos[0]
    dy = x[1] - anchor_pos[1]
    d = max(np.sqrt(dx ** 2 + dy ** 2), 0.1)
    # dh/dd = -10*n/(ln(10)*d)
    dh_dd = -10 * N_PLE / (np.log(10) * d)
    dd_dx = dx / d
    dd_dy = dy / d
    row = np.zeros(NX)
    row[0] = dh_dd * dd_dx
    row[1] = dh_dd * dd_dy
    return row


class EKF:
    def __init__(self, x0, P0=None):
        self.x = np.array(x0, dtype=float)
        self.P = np.eye(NX) * 1.0 if P0 is None else P0.copy()

    def predict(self, dt=DT):
        F = F_matrix(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def build_H_R(self, anchors, rssi_mask_effective, sigma_rssi):
        """anchors: (6,2). sigma_rssi: (6,) effective std per anchor (post NLOS handling).
        Returns H (8,6), R (8,8) for [6 rssi + 2 imu]."""
        H = np.zeros((8, NX))
        R = np.zeros(8)
        for i, a in enumerate(anchors):
            H[i, :] = H_rssi_row(self.x, a)
            R[i] = sigma_rssi[i] ** 2
        H[6, 2 + 2] = 1.0  # ax index=4
        H[7, 2 + 3] = 1.0  # ay index=5
        R[6] = 0.05 ** 2
        R[7] = 0.05 ** 2
        return H, np.diag(R)

    def predicted_z(self, anchors):
        z_hat = np.zeros(8)
        for i, a in enumerate(anchors):
            z_hat[i] = h_rssi(self.x, a)
        z_hat[6] = self.x[4]
        z_hat[7] = self.x[5]
        return z_hat

    def update(self, z, anchors, sigma_rssi, gate_thresh=None):
        """gate_thresh: if set, per-row normalized-innovation^2 > gate_thresh inflates that
        row's R (soft-reject outlier), preventing single bad draws from corrupting the state."""
        H, R = self.build_H_R(anchors, None, sigma_rssi)
        z_hat = self.predicted_z(anchors)
        y = z - z_hat  # innovation
        if gate_thresh is not None:
            S_diag = np.diag(H @ self.P @ H.T + R)
            nis_row = (y ** 2) / np.maximum(S_diag, 1e-6)
            bad = nis_row > gate_thresh
            if np.any(bad):
                R = R.copy()
                R[bad, bad] *= 1e4  # soft-reject: inflate variance, don't hard-drop
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(NX) - K @ H) @ self.P
        return y, S, H, R
