"""fig_rmse_boxplot.py -- RMSE boxplot, 4 methods, from mc_results.npz.
Data layout: each key in mc_results.npz is (N_trials, 3) = [RMSE, mean_NEES, avail_pct].
Adds Wilcoxon signed-rank significance brackets (Proposed VA vs. each baseline)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams.update({"font.family": "serif", "font.size": 9})

d = np.load("mc_results.npz")
methods = ["naive", "reject", "knn", "va"]
labels = ["Naive", "Reject", "k-NN", "Proposed VA"]
data = [d[m][:, 0] for m in methods]  # column 0 = RMSE

va_rmse = data[3]
pvals = {}
for i, name in enumerate(methods[:3]):
    _, p = stats.wilcoxon(va_rmse, data[i])
    pvals[name] = p

def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

fig, ax = plt.subplots(figsize=(3.6, 3.0))
bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("white")
ax.set_ylabel("RMSE (m)")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

ymax = max(np.max(v) for v in data)
step = ymax * 0.14
base = ymax * 1.05
for k, name in enumerate(methods[:3]):
    y = base + k * step
    x1, x2 = k + 1, 4
    ax.plot([x1, x1, x2, x2], [y, y + step * 0.15, y + step * 0.15, y], lw=0.8, c="black")
    ax.text((x1 + x2) / 2, y + step * 0.18, p_to_stars(pvals[name]), ha="center", va="bottom", fontsize=8)
ax.set_ylim(top=base + 3 * step + step)

plt.tight_layout()
plt.savefig("fig_rmse_boxplot.pdf")
plt.savefig("fig_rmse_boxplot.png", dpi=300)
print("saved fig_rmse_boxplot.pdf/png; p-values:", pvals)
