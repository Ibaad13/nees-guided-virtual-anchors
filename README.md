# NEES-Guided Virtual Anchor Geometry Recovery for RSSI-IMU Indoor Localization Under NLOS

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

**Repository for the paper draft and experimental results** on robust indoor localization using WiFi RSSI fused with IMU under Non-Line-of-Sight (NLOS) conditions.

The **implementation code is intentionally withheld** while the paper is being finalized. See [`Codes/README.md`](Codes/README.md) for details and expected release timeline.

---

## 📍 Problem Statement

Indoor localization using **WiFi RSSI** (Received Signal Strength Indicator) fused with **IMU** (Inertial Measurement Unit) data is cost-effective and widely deployable. However, it suffers significantly under **Non-Line-of-Sight (NLOS)** conditions — when walls or obstacles block the direct path between anchors and the target.

NLOS introduces two major issues:

1. **Systematic Bias** — Excess attenuation not captured by standard path-loss models leads to consistently overestimated distances.
2. **Geometric Degradation** — Dropping NLOS anchors can reduce usable measurements below the minimum required for 2D observability, forcing reliance on IMU dead-reckoning alone.

Traditional **Virtual Anchor (VA)** methods often rely on offline radio maps or machine learning. This work introduces a **model-based, online** approach that addresses a critical but previously under-discussed failure mode: **positive feedback loops** caused by generating VA pseudo-measurements from the filter's corrected state.

---

## 🧠 Method Overview

### 2.1 Process Model
State vector: $\mathbf{x} = [p_N, p_E, v_N, v_E]^{\mathsf{T}}$ (North-East planar position + velocity).

$$
\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_k + \mathbf{G}\mathbf{u}_k + \mathbf{w}_k
$$

where
$$
\mathbf{F} = \begin{bmatrix} \mathbf{I}_2 & \Delta t\,\mathbf{I}_2 \\ \mathbf{0}_2 & \mathbf{I}_2 \end{bmatrix}, \quad
\mathbf{G} = \begin{bmatrix} \tfrac{1}{2}\Delta t^2\mathbf{I}_2 \\ \Delta t\,\mathbf{I}_2 \end{bmatrix}
$$

### 2.2 Measurement Models
**RSSI (Log-distance path-loss):**
$$
z_{i,k} = \mathrm{PL}_0 - 10n\log_{10}\!\left(\frac{d_{i,k}}{d_0}\right) + v_{i,k}
$$

**IMU:** Direct acceleration measurements with appropriate noise modeling.

### 2.3 Three-Layer Architecture (Core Contribution)

**Layer 1 — NLOS Detection**  
Per-anchor Normalized Estimation Error Squared (NEES) with sliding-window voting.

**Layer 2 — GDOP-Optimal Virtual Anchor Placement**  
Places a synthetic anchor at the location that maximally restores geometric observability (minimizes trace of the inverse Fisher Information matrix).

**Layer 3 — Safety Mechanism**  
- Uses exponentially-smoothed position for VA generation  
- Weaker covariance for VA measurements  
- Dual Normalized Innovation Squared (NIS) gating to prevent divergent updates

This architecture explicitly **breaks the positive feedback loop** identified in naive VA implementations.

---

## 🔬 Simulation Setup

| Parameter              | Value          | Unit     |
|------------------------|----------------|----------|
| Hall dimensions        | 20 × 15        | m        |
| Real anchors           | 6              | —        |
| NLOS walls             | 3              | —        |
| Trajectory duration    | 120            | s        |
| IMU Δt                 | 0.1            | s        |
| Path-loss exponent $n$ | 2.5            | —        |
| Monte Carlo trials     | **50**         | —        |

**Compared Methods:**
- **Naive** — No NLOS handling
- **Reject** — Drop flagged anchors
- **k-NN NLOS** — Lightweight ML-style outlier detection baseline
- **Proposed VA** — NEES-guided + GDOP-optimal + Safety layer

---

## 📊 Results

### Headline Performance

| Method         | RMSE Mean (m) | RMSE Std (m) | Mean NEES | Avail. < 3 Anch. |
|----------------|---------------|--------------|-----------|------------------|
| Naive          | 3.56          | 0.09         | 5.48      | 0.00%            |
| Reject         | 3.61          | 3.00         | 7.70      | 6.51%            |
| k-NN NLOS      | 3.93          | 0.13         | 5.68      | 1.59%            |
| **Proposed VA**| **1.20**      | **1.08**     | **6.90**  | **0.00%**        |

**Key Achievement:** **66–67% RMSE reduction** with zero loss in anchor availability and strong consistency.

### Additional Insights
- **k-NN baseline** underperformed naive (both in accuracy and compute cost).
- Safety layer prevented problematic updates in ~8.3% of flagged steps.
- All methods run well under 1 ms/step (real-time capable).

---

## 📈 Figures

All figures are available as high-quality vector PDFs in [`Results/Figures/`](Results/Figures/):

- **`fig_environment.pdf`** — Environment layout with anchors, walls, trajectory, and spawned VAs
- **`fig_rmse_boxplot.pdf`** — RMSE distribution across 50 trials
- **`fig_cdf.pdf`** — Cumulative distribution of RMSE
- **`fig_nees_timeline.pdf`** — NEES timeline + position error comparison
- **`fig_computational.pdf`** — Runtime comparison

Raw Monte Carlo results: [`Results/mc_results.npz`](Results/mc_results.npz)

---

## ⚠️ Scope & Honesty Notes

- Simulation-only (synthetic hall and trajectory)
- Paper draft in progress
- Novelty focused on feedback-loop analysis and NEES-guided recovery
- All limitations are explicitly stated in the accompanying write-up

---

## 📜 On Priority & Reuse

This repository provides a **dated, public record** of the work. The standard academic priority mechanisms apply:

1. GitHub commit history
2. Upcoming preprint (arXiv)
3. Formal citation (will be added once DOI/arXiv ID is available)

**Feel free to build upon the ideas** — research thrives on iteration. Please cite appropriately if you use results, figures, or concepts from this work.

If you're a researcher/reviewer and need early code access for evaluation, please reach out.

---

## 👤 Contact

**Muhammad Ibaad**  
Dawood University of Engineering and Technology, Karachi, Pakistan  
📧 ibaadsajidshaikh18@gmail.com

---

**A preprint is in progress.** Citation details will be added upon availability.

---

*Made with ❤️ for reproducible and transparent research.*
