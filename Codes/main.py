"""Full pipeline, in order:
  1. Environment/trajectory demo figure
  2-4. Single-trial demo runs (naive / reject / VA) with their own figures
  5. Full 50-trial x 4-method (naive/reject/knn/va) Monte Carlo -> mc_results.npz
  6. Text + LaTeX summary of Monte Carlo results
  7-11. Final IEEE-styled paper figures (PDF+PNG), built from mc_results.npz
        and from fresh single-trial runs where needed
Run this top-level script to regenerate everything from scratch.
"""
import subprocess
import sys

STEPS = [
    "trajectory.py",          # fig_env.png (demo)
    "run_baseline.py",        # fig_ekf_baseline.png (demo)
    "run_nees.py",             # fig_nees.png (demo)
    "run_va.py",                # fig_va.png (demo)
    "monte_carlo.py",         # mc_results.npz (50 trials x naive/reject/knn/va)
    "analysis.py",             # console: text summary + LaTeX table
    "fig_environment.py",     # fig_environment.pdf/.png  (paper Fig. 1)
    "fig_rmse_boxplot.py",    # fig_rmse_boxplot.pdf/.png (paper Fig. 2)
    "fig_cdf.py",               # fig_cdf.pdf/.png            (paper Fig. 3)
    "fig_nees_timeline.py",   # fig_nees_timeline.pdf/.png (paper Fig. 4)
    "fig_computational.py",   # fig_computational.pdf/.png (paper Fig. 5)
]


def main():
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        result = subprocess.run([sys.executable, step], capture_output=False)
        if result.returncode != 0:
            print(f"FAILED at {step} (exit {result.returncode})")
            sys.exit(1)
    print("\n=== Pipeline complete. ===")
    print("Demo figures: fig_env.png, fig_ekf_baseline.png, fig_nees.png, fig_va.png")
    print("Monte Carlo data: mc_results.npz")
    print("Final paper figures (PDF+PNG): fig_environment.*, fig_rmse_boxplot.*,")
    print("  fig_cdf.*, fig_nees_timeline.*, fig_computational.*")


if __name__ == "__main__":
    main()
