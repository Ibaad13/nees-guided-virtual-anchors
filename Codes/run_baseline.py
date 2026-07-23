"""Baseline EKF run: naive (all anchors, LOS-only sigma model unaware of NLOS)."""
import numpy as np
import matplotlib.pyplot as plt
from env import ANCHORS, WALLS, HALL_W, HALL_H
from trajectory import generate_trajectory, DT, N
from sensors import generate_rssi, generate_imu, SIGMA_LOS
from ekf import EKF


def run_baseline(seed=0):
    rng = np.random.default_rng(seed)
    traj = generate_trajectory(seed=seed)

    x0 = np.array([traj["px_true"][0], traj["py_true"][0], traj["vx"][0], traj["vy"][0], 0.0, 0.0])
    ekf = EKF(x0)

    est = np.zeros((N, 6))
    bias_state = np.zeros(2)
    for k in range(N):
        if k > 0:
            ekf.predict(DT)
        tag_true = np.array([traj["px_true"][k], traj["py_true"][k]])
        rssi, mask = generate_rssi(ANCHORS, tag_true, WALLS, rng)

        vel_prev = np.array([traj["vx"][max(k - 1, 0)], traj["vy"][max(k - 1, 0)]])
        vel_curr = np.array([traj["vx"][k], traj["vy"][k]])
        acc, bias_state = generate_imu(vel_prev, vel_curr, DT, bias_state, rng)

        z = np.concatenate([rssi, acc])
        sigma_rssi_naive = np.full(6, SIGMA_LOS)  # naive: assumes LOS always
        ekf.update(z, ANCHORS, sigma_rssi_naive)
        est[k] = ekf.x

    return traj, est


if __name__ == "__main__":
    traj, est = run_baseline()
    rmse = np.sqrt(np.mean((est[:, 0] - traj["px_true"]) ** 2 + (est[:, 1] - traj["py_true"]) ** 2))
    print(f"Naive EKF RMSE: {rmse:.3f} m")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, HALL_W)
    ax.set_ylim(0, HALL_H)
    ax.plot(traj["px_true"], traj["py_true"], color="black", linewidth=1.5, label="Ground truth")
    ax.plot(est[:, 0], est[:, 1], color="red", linewidth=1.0, alpha=0.8, label="EKF estimate")
    ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c="green", marker="^", s=80, label="Anchors")
    for seg in WALLS:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Baseline EKF (Naive) RMSE={rmse:.2f}m")
    plt.savefig("fig_ekf_baseline.png", dpi=150)
    print("saved fig_ekf_baseline.png")
