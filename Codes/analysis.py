"""Text/LaTeX summary of Monte Carlo results (all 4 methods).
Plotting is handled by the dedicated fig_*.py scripts (IEEE-styled, PDF+PNG,
used directly in the paper) -- this module intentionally does NOT also
generate fig_rmse_boxplot/fig_cdf images, to avoid two different-looking
versions of the same-named figure existing in the project."""
import numpy as np

METHOD_ORDER = ["naive", "reject", "knn", "va"]
METHOD_LABELS = {"naive": "Naive", "reject": "Reject", "knn": "k-NN NLOS", "va": "VA (NEES-guided)"}


def load():
    d = np.load("mc_results.npz")
    return {k: d[k] for k in d.files}


def summarize(res):
    stats = {}
    for m in METHOD_ORDER:
        if m not in res:
            continue
        r = res[m]
        stats[m] = {
            "rmse_mean": r[:, 0].mean(), "rmse_std": r[:, 0].std(),
            "rmse_median": np.median(r[:, 0]),
            "nees_mean": r[:, 1].mean(),
            "avail_mean": r[:, 2].mean(),
        }
    return stats


def print_latex_table(stats):
    print("\n% LaTeX table\n")
    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{Monte Carlo Results (N=50 trials)}")
    print(r"\label{tab:mc_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & RMSE mean (m) & RMSE std (m) & Mean NEES $\bar\epsilon$ & Avail.\ $<3$ anch.\ (\%) \\")
    print(r"\midrule")
    for m in METHOD_ORDER:
        if m not in stats:
            continue
        s = stats[m]
        print(f"{METHOD_LABELS[m]} & {s['rmse_mean']:.2f} & {s['rmse_std']:.2f} & "
              f"{s['nees_mean']:.2f} & {s['avail_mean']:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    res = load()
    stats = summarize(res)
    for m in METHOD_ORDER:
        if m not in stats:
            continue
        s = stats[m]
        print(f"{m}: RMSE mean={s['rmse_mean']:.2f} median={s['rmse_median']:.2f} std={s['rmse_std']:.2f} "
              f"NEES={s['nees_mean']:.2f} avail={s['avail_mean']:.2f}%")
    print_latex_table(stats)
    print("\n(Plots are generated separately by fig_rmse_boxplot.py, fig_cdf.py, etc.)")
