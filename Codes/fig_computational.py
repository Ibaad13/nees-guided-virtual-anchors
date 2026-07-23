"""fig_computational.py -- per-step runtime (ms), 4 methods. Bars use MEASURED
values from Sec. V-C (Python/NumPy, single core, mean over 50 trials x 1200 steps,
warm-up-excluded); see monte_carlo timing instrumentation for the measurement script.
Hardcoded here (not re-measured on every figure build) to keep figure generation fast
and reproducible; update these constants if the underlying implementation changes."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.family": "serif", "font.size": 9})

methods = ["Naive", "Reject", "k-NN", "Proposed VA"]
# measured mean per-step time (ms), see Sec. V-C
times_ms = [0.147, 0.150, 0.934, 0.476]
colors = ["gray", "tab:orange", "tab:purple", "tab:green"]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.bar(methods, times_ms, color=colors)
ax.set_ylabel("Time per step (ms)")
ax.set_ylim(0, max(times_ms) * 1.5)
for b, v in zip(bars, times_ms):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
plt.tight_layout()
plt.savefig("fig_computational.pdf")
plt.savefig("fig_computational.png", dpi=300)
print("saved fig_computational.pdf")
# Note: 10 Hz budget (100 ms) omitted from plot -- all methods are ~2 orders of
# magnitude below it, so a 100 ms reference line would be off-scale/uninformative
# at this y-range. Mention the budget in text (Sec. V-C) instead.
