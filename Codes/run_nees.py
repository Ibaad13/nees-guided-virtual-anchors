"""Run EKF w/ per-anchor NEES detector; plot NEES time-series + NLOS highlights."""
import numpy as np
import matplotlib.pyplot as plt
from env import ANCHORS, WALLS
from trajectory import generate_trajectory, DT, N
from sensors import generate_rssi, generate_imu
from ekf_nees import EKF_NEES
from nees_detector import CHI2_THRESH_1DOF_95


def run_nees(seed=0, mode="reject"):
    rng = np.random.default_rng(seed)
    traj = generate_trajectory(seed=seed)
    x0 = np.array([traj["px_true"][0], traj["py_true"][0], traj["vx"][0], traj["vy"][0], 0.0, 0.0])
    ekf = EKF_NEES(x0)

    est = np.zeros((N, 6))
    eps_hist = np.zeros((N, 6))
    flags_hist = np.zeros((N, 6), dtype=bool)
    true_nlos_hist = np.zeros((N, 6), dtype=bool)
    bias_state = np.zeros(2)

    for k in range(N):
        tag_true = np.array([traj["px_true"][k], traj["py_true"][k]])
        rssi, true_mask = generate_rssi(ANCHORS, tag_true, WALLS, rng)  # true_mask: True=LOS
        vel_prev = np.array([traj["vx"][max(k - 1, 0)], traj["vy"][max(k - 1, 0)]])
        vel_curr = np.array([traj["vx"][k], traj["vy"][k]])
        acc, bias_state = generate_imu(vel_prev, vel_curr, DT, bias_state, rng)
        z = np.concatenate([rssi, acc])

        flags, eps = ekf.step(z, ANCHORS, DT, mode=mode)
        est[k] = ekf.x
        eps_hist[k] = eps
        flags_hist[k] = flags
        true_nlos_hist[k] = ~true_mask

    return traj, est, eps_hist, flags_hist, true_nlos_hist


if __name__ == "__main__":
    traj, est, eps_hist, flags_hist, true_nlos_hist = run_nees()
    rmse = np.sqrt(np.mean((est[:, 0] - traj["px_true"]) ** 2 + (est[:, 1] - traj["py_true"]) ** 2))
    print(f"Reject-mode EKF RMSE: {rmse:.3f} m")

    t = traj["t"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        ax = axes[i]
        ax.plot(t, eps_hist[:, i], color="tab:blue", linewidth=0.7, label="NEES $\\epsilon_i$")
        ax.axhline(CHI2_THRESH_1DOF_95, color="gray", linestyle="--", linewidth=0.8, label="$\\chi^2_{0.95,1}$")
        # highlight detected NLOS regions
        ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                         where=flags_hist[:, i], color="red", alpha=0.2, transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"Anchor {i}")
        ax.set_ylim(0, min(np.percentile(eps_hist[:, i], 99) + 5, 100))
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Per-Anchor NEES Time-Series (red = detected NLOS)")
    plt.tight_layout()
    plt.savefig("fig_nees.png", dpi=150)
    print("saved fig_nees.png")

    # detection quality summary
    det_flat = flags_hist.flatten()
    true_flat = true_nlos_hist.flatten()
    tp = np.sum(det_flat & true_flat)
    fp = np.sum(det_flat & ~true_flat)
    fn = np.sum(~det_flat & true_flat)
    tn = np.sum(~det_flat & ~true_flat)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"NLOS detection: precision={precision:.3f} recall={recall:.3f} (TP={tp} FP={fp} FN={fn} TN={tn})")
