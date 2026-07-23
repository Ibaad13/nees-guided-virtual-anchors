"""Run EKF w/ NEES-guided VA replacement; plot real anchors, NLOS anchor, VA, trajectory."""
import numpy as np
import matplotlib.pyplot as plt
from env import ANCHORS, WALLS, HALL_W, HALL_H
from trajectory import generate_trajectory, DT, N
from sensors import generate_rssi, generate_imu
from ekf_va import EKF_VA


def run_va(seed=0):
    rng = np.random.default_rng(seed)
    traj = generate_trajectory(seed=seed)
    x0 = np.array([traj["px_true"][0], traj["py_true"][0], traj["vx"][0], traj["vy"][0], 0.0, 0.0])
    ekf = EKF_VA(x0)

    est = np.zeros((N, 6))
    flags_hist = np.zeros((N, 6), dtype=bool)
    va_pos_hist = []
    bias_state = np.zeros(2)

    for k in range(N):
        tag_true = np.array([traj["px_true"][k], traj["py_true"][k]])
        rssi, true_mask = generate_rssi(ANCHORS, tag_true, WALLS, rng)
        vel_prev = np.array([traj["vx"][max(k - 1, 0)], traj["vy"][max(k - 1, 0)]])
        vel_curr = np.array([traj["vx"][k], traj["vy"][k]])
        acc, bias_state = generate_imu(vel_prev, vel_curr, DT, bias_state, rng)

        flags, eps, va_pos = ekf.step(rssi, acc, ANCHORS, rng, DT)
        est[k] = ekf.x
        flags_hist[k] = flags
        va_pos_hist.append(va_pos)

    return traj, est, flags_hist, va_pos_hist


if __name__ == "__main__":
    traj, est, flags_hist, va_pos_hist = run_va()
    rmse = np.sqrt(np.mean((est[:, 0] - traj["px_true"]) ** 2 + (est[:, 1] - traj["py_true"]) ** 2))
    print(f"VA-mode EKF RMSE: {rmse:.3f} m")

    chosen_k = None
    for k in range(N // 4, N):
        if flags_hist[k].sum() == 1:
            chosen_k = k
            break
    if chosen_k is None:
        chosen_k = int(np.argmax(flags_hist.sum(axis=1)))

    nlos_idx = np.where(flags_hist[chosen_k])[0]
    va_snapshot = va_pos_hist[chosen_k]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, HALL_W)
    ax.set_ylim(0, HALL_H)
    ax.plot(traj["px_true"], traj["py_true"], color="black", linewidth=1.0, alpha=0.4, label="Trajectory")
    los_idx = [i for i in range(6) if i not in nlos_idx]
    ax.scatter(ANCHORS[los_idx, 0], ANCHORS[los_idx, 1], c="green", marker="^", s=100, label="Real anchor (LOS)")
    if len(nlos_idx) > 0:
        ax.scatter(ANCHORS[nlos_idx, 0], ANCHORS[nlos_idx, 1], c="red", marker="^", s=100, label="Real anchor (NLOS)")
        for i in nlos_idx:
            vp = va_snapshot[i]
            if vp is not None:
                ax.scatter(vp[0], vp[1], c="blue", marker="*", s=200,
                           label="Virtual Anchor" if i == nlos_idx[0] else None)
                ax.plot([ANCHORS[i, 0], vp[0]], [ANCHORS[i, 1], vp[1]], "b--", linewidth=0.8, alpha=0.6)
    ax.scatter(traj["px_true"][chosen_k], traj["py_true"][chosen_k], c="orange", marker="o", s=80,
               label="Tag @ snapshot", zorder=5)
    for seg in WALLS:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"VA Spawn Snapshot (t={traj['t'][chosen_k]:.1f}s), RMSE={rmse:.2f}m")
    plt.savefig("fig_va.png", dpi=150)
    print(f"saved fig_va.png (snapshot k={chosen_k}, nlos_anchors={list(nlos_idx)})")
