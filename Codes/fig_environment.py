"""fig_environment.py -- hall, anchors, walls, trajectory, VA locations (one trial)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.family": "serif", "font.size": 9})

from env import ANCHORS, WALLS, HALL_W, HALL_H
from run_va import run_va

traj, est, flags_hist, va_pos_hist = run_va(seed=0)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.set_xlim(0, HALL_W)
ax.set_ylim(0, HALL_H)
ax.plot(traj["px_true"], traj["py_true"], color="tab:blue", linewidth=1.0, label="True trajectory")
ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c="green", marker="o", s=40, label="Real anchors", zorder=5)
for seg in WALLS:
    (x1, y1), (x2, y2) = seg
    ax.plot([x1, x2], [y1, y2], color="gray", linewidth=3, solid_capstyle="round")

# collect unique VA positions used across the trial
va_pts = set()
for step_va in va_pos_hist:
    for i, pos in step_va.items():
        if pos is not None:
            va_pts.add((round(float(pos[0]), 2), round(float(pos[1]), 2)))
if va_pts:
    va_arr = np.array(list(va_pts))
    ax.scatter(va_arr[:, 0], va_arr[:, 1], c="red", marker="*", s=60,
               label="VA locations", zorder=6)

ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.legend(fontsize=6, loc="upper right")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("fig_environment.pdf")
plt.savefig("fig_environment.png", dpi=300)
print("saved fig_environment.pdf")
