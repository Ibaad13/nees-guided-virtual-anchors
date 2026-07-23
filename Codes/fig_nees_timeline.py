"""fig_nees_timeline.py -- representative trial: per-anchor NEES timeline (top) and
position error norm over time for naive/reject/proposed VA (bottom), shared x-axis."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.family": "serif", "font.size": 8})

from run_baseline import run_baseline
from run_nees import run_nees
from run_va import run_va
from nees_detector import CHI2_THRESH_1DOF_95

SEED = 3  # representative trial with clear NLOS episodes

traj_n, est_n = run_baseline(seed=SEED)
traj_r, est_r, eps_hist, flags_hist, _ = run_nees(seed=SEED, mode="reject")
traj_v, est_v, flags_va_hist, va_pos_hist = run_va(seed=SEED)

t = traj_n["t"]
err_naive = np.sqrt((est_n[:, 0] - traj_n["px_true"]) ** 2 + (est_n[:, 1] - traj_n["py_true"]) ** 2)
err_reject = np.sqrt((est_r[:, 0] - traj_r["px_true"]) ** 2 + (est_r[:, 1] - traj_r["py_true"]) ** 2)
err_va = np.sqrt((est_v[:, 0] - traj_v["px_true"]) ** 2 + (est_v[:, 1] - traj_v["py_true"]) ** 2)

fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

ax0 = axes[0]
any_flag = flags_hist.any(axis=1)
ax0.fill_between(t, 0, 1, where=any_flag, transform=ax0.get_xaxis_transform(),
                  color="red", alpha=0.12, label="NLOS (any anchor)")
for i in range(6):
    ax0.plot(t, np.clip(eps_hist[:, i], 0, 60), linewidth=0.4, alpha=0.7)
ax0.axhline(CHI2_THRESH_1DOF_95, color="black", linestyle="--", linewidth=0.7,
            label=r"$\chi^2_{0.95,1}$")
va_activation = flags_va_hist.any(axis=1)
act_times = t[va_activation]
if len(act_times) > 0:
    # mark onset of each contiguous VA-active segment
    onsets = act_times[np.r_[True, np.diff(act_times) > 0.15]]
    for ot in onsets:
        ax0.annotate("", xy=(ot, 55), xytext=(ot, 65),
                     arrowprops=dict(arrowstyle="->", color="green", lw=1))
ax0.set_ylabel("Per-anchor NEES")
ax0.set_ylim(0, 70)
ax0.legend(fontsize=6, loc="upper right")

ax1 = axes[1]
ax1.plot(t, err_naive, color="gray", linewidth=0.8, label="Naive")
ax1.plot(t, err_reject, color="tab:orange", linewidth=0.8, label="Reject")
ax1.plot(t, err_va, color="tab:green", linewidth=0.8, label="Proposed VA")
ax1.set_ylabel("Position error (m)")
ax1.set_xlabel("Time (s)")
ax1.legend(fontsize=6, loc="upper right")

plt.tight_layout()
plt.savefig("fig_nees_timeline.pdf")
plt.savefig("fig_nees_timeline.png", dpi=300)
print("saved fig_nees_timeline.pdf")
