"""Synthetic ground-truth trajectory generator."""
import numpy as np
from env import HALL_W, HALL_H

DT = 0.1
T_TOTAL = 120.0
N = int(T_TOTAL / DT)


def generate_trajectory(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT

    # Waypoint path with 3 turns inside hall bounds (margin 1.5m)
    wp = np.array([
        [2.0, 2.0],
        [17.0, 2.0],
        [17.0, 7.5],
        [10.0, 13.0],
        [2.0, 13.0],
    ])
    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    seg_frac = seg_len / seg_len.sum()
    seg_samples = np.round(seg_frac * N).astype(int)
    seg_samples[-1] = N - seg_samples[:-1].sum()

    pos_list = []
    for i in range(len(wp) - 1):
        n_i = seg_samples[i]
        alpha = np.linspace(0, 1, n_i, endpoint=False)[:, None]
        seg_pos = wp[i][None, :] + alpha * (wp[i + 1] - wp[i])[None, :]
        pos_list.append(seg_pos)
    pos = np.vstack(pos_list)
    if pos.shape[0] < N:
        pos = np.vstack([pos, np.tile(wp[-1], (N - pos.shape[0], 1))])
    pos = pos[:N]

    vel = np.gradient(pos, DT, axis=0)

    # add small gaussian noise to positions (sensor-realism, not used as truth for EKF eval directly)
    pos_noisy = pos + rng.normal(0, 0.01, pos.shape)

    return {
        "t": t,
        "px": pos_noisy[:, 0], "py": pos_noisy[:, 1],
        "vx": vel[:, 0], "vy": vel[:, 1],
        "px_true": pos[:, 0], "py_true": pos[:, 1],
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from env import ANCHORS, WALLS

    traj = generate_trajectory()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, HALL_W)
    ax.set_ylim(0, HALL_H)
    ax.plot(traj["px"], traj["py"], color="blue", linewidth=1.5, label="Trajectory")
    ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c="green", marker="^", s=80, label="Anchors")
    for seg in WALLS:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Environment + Trajectory")
    plt.savefig("fig_env.png", dpi=150)
    print("saved fig_env.png")
