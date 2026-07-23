"""fig_cdf.py -- CDF of per-trial RMSE for 4 methods, from mc_results.npz."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.family": "serif", "font.size": 9})

d = np.load("mc_results.npz")
methods = ["naive", "reject", "knn", "va"]
labels = ["Naive", "Reject", "k-NN", "Proposed VA"]
colors = ["gray", "tab:orange", "tab:purple", "tab:green"]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for m, lab, c in zip(methods, labels, colors):
    vals = np.sort(d[m][:, 0])
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, label=lab, color=c)
ax.set_xlabel("Error (m)")
ax.set_ylabel("CDF")
ax.set_xlim(0, 15)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig_cdf.pdf")
plt.savefig("fig_cdf.png", dpi=300)
print("saved fig_cdf.pdf")
