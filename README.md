<!--
###############################################################################
# NEES-Guided Virtual Anchor Geometry Recovery for RSSI-IMU Indoor Localization
# 
# This README is the public-facing entry point for the project repository.
# It contains all results, figures, and a detailed technical description.
# The implementation code is temporarily withheld (see Codes/README.md).
#
# Last updated: 2026-07-12
###############################################################################
-->

<p align="center">
  <h1 align="center">📍 NEES-Guided Virtual Anchor Geometry Recovery<br>for RSSI-IMU Indoor Localization Under NLOS</h1>
  <p align="center">
    <strong>Status:</strong> <em>work in progress</em> · 
    <a href="https://github.com/yourusername/yourrepo/issues">Report Bug</a> ·
    <a href="https://github.com/yourusername/yourrepo/pulls">Request Feature</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/status-WIP-yellow?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/language-MATLAB/Python-blue?style=flat-square" alt="Language">
    <img src="https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square" alt="License">
    <img src="https://img.shields.io/badge/paper-in%20progress-orange?style=flat-square" alt="Paper">
    <img src="https://img.shields.io/badge/arXiv-pending-brightgreen?style=flat-square" alt="arXiv">
  </p>
</p>

---

## 📖 Table of Contents

- [Problem](#-problem)
- [Method and Mathematics](#-method-and-mathematics)
- [Simulation Setup](#-simulation-setup)
- [Results](#-results)
- [Scope, Honesty Notes, and What's Not Claimed](#-scope-honesty-notes-and-whats-not-claimed)
- [On Priority and Reuse of This Work](#-on-priority-and-reuse-of-this-work)
- [Contact](#-contact)
- [Figures](#-figures)

---

## 🚨 Problem

Indoor localization using WiFi RSSI fused with IMU data is cheap and ubiquitous, but suffers badly under **non-line-of-sight (NLOS)** propagation — when a wall or other obstruction sits between an anchor and the tag being tracked. NLOS does two separate kinds of damage:

1. **Bias** — the RSSI‑to‑distance model assumes free‑space‑like decay; NLOS adds a large, roughly one‑sided excess attenuation that the model doesn't account for, so range estimates from that anchor are systematically wrong, not just noisier.
2. **Geometric weakness** — the standard fix (drop/down‑weight anchors flagged as NLOS) removes information. Once fewer than three anchors remain usable, 2‑D position becomes **formally unobservable** from range measurements alone (see the Fisher‑information argument in §2.3 below) — the filter is coasting on IMU dead‑reckoning alone until an anchor comes back into view.

**Existing "virtual anchor" (VA) approaches** — placing a synthetic anchor to patch the geometry — mostly come from machine‑learning fingerprint prediction or offline radio‑map interpolation, computed ahead of time from a signal map, not in response to a live NLOS event. A **model‑based** alternative (predict a synthetic anchor's measurement from the filter's own propagation model, no signal map needed) sounds simpler but has a **specific, previously unaddressed failure mode**: if that synthetic measurement is generated from the filter's *corrected* state (the state produced by the very update the VA feeds into), a state error can be **reinforced by the very measurement meant to correct it** — a positive feedback loop, not a recovery mechanism.

This project is about **(a)** formally identifying that feedback loop, **(b)** building a mechanism that avoids it, and **(c)** validating that mechanism with real Monte Carlo numbers rather than assuming it works.

---

## 🧮 Method and Mathematics

<details>
<summary><strong>Click to expand the mathematical details</strong></summary>

### Process model

State vector: $\mathbf{x} = [p_N, p_E, v_N, v_E]^{\mathsf{T}}$ (planar position + velocity, north‑east frame). Driven by IMU accelerometer input $\mathbf{u}_k = [a_N, a_E]^{\mathsf{T}}$:

$$
\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_k + \mathbf{G}\mathbf{u}_k + \mathbf{w}_k
$$

$$
\mathbf{F} = \begin{bmatrix} \mathbf{I}_2 & \Delta t\,\mathbf{I}_2 \\ \mathbf{0}_2 & \mathbf{I}_2 \end{bmatrix}, \quad
\mathbf{G} = \begin{bmatrix} \tfrac{1}{2}\Delta t^2\mathbf{I}_2 \\ \Delta t\,\mathbf{I}_2 \end{bmatrix}
$$

with process noise $\mathbf{Q} = \mathrm{diag}(0.01, 0.01, 0.05, 0.05)$.

The model is deliberately planar. Roll/pitch/yaw are assumed supplied externally (AHRS/magnetometer) and used to rotate raw body‑frame accelerometer readings into this horizontal frame before they enter the filter — a standard loosely‑coupled simplification, stated explicitly rather than left implicit. A full 3‑D deployment would instead carry attitude in the state itself.

### Measurement models

**RSSI** (log‑distance path‑loss model), anchor $i$:

$$
z_{i,k} = \mathrm{PL}_0 - 10n\log_{10}\!\left(\frac{d_{i,k}}{d_0}\right) + v_{i,k}
$$

where $d_{i,k} = \|\mathbf{p}_k - \mathbf{a}_i\|$, $n = 2.5$ (path‑loss exponent), $\mathrm{PL}_0 = -40$ dB at $d_0 = 1$ m, and $v_{i,k} \sim \mathcal{N}(0, \sigma_{\mathrm{LOS}}^2)$ with $\sigma_{\mathrm{LOS}} = 2$ dB. Under NLOS, variance increases: $R_{i,k} = \sigma_{\mathrm{LOS}}^2 + \sigma_{\mathrm{NLOS}}^2$ with $\sigma_{\mathrm{NLOS}} \sim \mathcal{N}(8, 4^2)$ dB.

**IMU:** $\mathbf{z}_{\mathrm{IMU},k} = [a_N, a_E]^{\mathsf{T}} + \boldsymbol{\eta}_k$, $\boldsymbol{\eta}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{\mathrm{IMU}})$.

### Why fewer than 3 anchors breaks observability (Fisher information)

For $M$ anchors in LOS, the position Fisher information matrix under the log‑distance model is

$$
\mathbf{J} = \sum_{i=1}^{M} \frac{1}{\sigma_i^2}\left(\frac{10n}{d_i \ln 10}\right)^{2} \mathbf{r}_i \mathbf{r}_i^{\mathsf{T}}
$$

where $\mathbf{r}_i = (\mathbf{p} - \mathbf{a}_i)/d_i$ is the unit direction to anchor $i$. This is the same quantity (up to weighting) whose inverse trace defines GDOP below. When $M < 3$ anchors remain in LOS — or the survivors are near‑collinear with the tag — $\mathrm{rank}(\mathbf{J}) < 2$, $\mathbf{J}$ is singular, and GDOP diverges: position is no longer observable from range measurements alone. **This is the exact failure that anchor rejection can induce, and that GDOP‑optimal VA placement is designed to reverse:** adding a VA row to $\mathbf{J}$ restores rank whenever its direction is linearly independent of the surviving real anchors.

### The feedback‑loop problem (the core issue this project addresses)

If a VA pseudo‑measurement is generated from the **corrected** state $\hat{\mathbf{x}}_{k|k}$ rather than the prediction, its innovation at the next step is:

$$
\boldsymbol{\nu}_{\mathrm{VA}} = h(\mathbf{c}^*, \hat{\mathbf{x}}_{k|k}) - h(\mathbf{c}^*, \hat{\mathbf{x}}_{k|k-1}) + \boldsymbol{\eta}
\approx \mathbf{H}\mathbf{K}\boldsymbol{\nu}_{k-1} + \boldsymbol{\eta}
$$

since $\hat{\mathbf{x}}_{k|k} - \hat{\mathbf{x}}_{k|k-1} = \mathbf{K}\boldsymbol{\nu}_{k-1}$ (the Kalman update itself). This shows the current VA innovation is a function of the **previous update's innovation**, not of the true state error — if the previous update happened to move the estimate the wrong way, the VA tends to *confirm* that move instead of correcting it. That's a positive feedback loop hiding inside what looks like a correction mechanism.

### Three‑layer architecture (the fix)

**Layer 1 — Detection (per‑anchor NEES).** For anchor $i$ at time $k$:

$$
\epsilon_{i,k} = \boldsymbol{\nu}_{i,k}^{\mathsf{T}} S_{i,k}^{-1} \boldsymbol{\nu}_{i,k} \sim \chi^2_1 \text{ (under LOS)}
$$

NLOS is flagged when $\epsilon_{i,k} > \gamma = 3.84$ (95% confidence) for at least 3 of the last 5 steps — a sliding‑window vote to suppress false alarms from ordinary multipath fading.

**Layer 2 — Placement (GDOP‑optimal VA).** When anchor $i$ is flagged, a VA is placed at:

$$
\mathbf{c}^* = \arg\min_{\mathbf{c}_j} \sqrt{\mathrm{tr}\!\left((\mathbf{H}_{\mathrm{geo},j}^{\mathsf{T}}\mathbf{H}_{\mathrm{geo},j})^{-1}\right)}
$$

searched over a grid of the remaining hall area (candidates too close to the current position estimate are excluded — near that range, small position error maps to large RSSI error). Minimizing this is equivalent to maximizing the Fisher information the candidate adds (§2.3), so the chosen VA is the placement doing the most to restore $\mathrm{rank}(\mathbf{J})$.

**Layer 3 — Safety (breaks the loop from §2.4).** The VA pseudo‑measurement is generated from an **exponentially‑smoothed** position estimate, $\bar{\mathbf{p}}_k = (1-\alpha)\bar{\mathbf{p}}_{k-1} + \alpha\hat{\mathbf{p}}_{k|k}$, not the raw corrected or predicted state — decoupling it further from the short‑timescale correlation identified above. It's assigned a weaker covariance ($\sigma_{\mathrm{VA}}^2 > \sigma_{\mathrm{LOS}}^2$) than a real LOS anchor, and before any update is applied, both a per‑row and a whole‑vector normalized‑innovation‑squared (NIS) test run; either failing skips that measurement (or the whole step) rather than letting it inject a large, potentially divergent correction.

</details>

---

## 🧪 Simulation Setup

| Parameter | Value | Unit |
|-----------|-------|------|
| Hall dimensions | 20 × 15 | m |
| Real anchors | 6 | — |
| NLOS walls | 3 | — |
| Trajectory duration | 120 | s |
| IMU Δt | 0.1 | s |
| Path‑loss exponent $n$ | 2.5 | — |
| $PL_0$ at $d_0=1$ m | −40 | dB |
| RSSI noise (LOS) | $\mathcal{N}(0, 2^2)$ | dB |
| RSSI excess (NLOS) | $\mathcal{N}(8, 4^2)$ | dB |
| Process noise $\sigma_p$ | 0.05 | m |
| Process noise $\sigma_v$ | 0.3 | m/s |
| GDOP grid resolution | 0.5 | m |

Four methods compared over **N = 50 Monte Carlo trials**:
- **Naive** — all anchors used, no NLOS awareness.
- **Reject** — flagged anchors get $R_i = 10^6$ (effectively dropped).
- **k‑NN NLOS** — a training‑free local‑outlier‑factor‑style baseline: each anchor's RSSI is checked against a rolling 20‑sample window using $k$‑nearest‑neighbor distance; flagged anchors are rejected the same way as the Reject method. Included as a lightweight‑ML comparator, distinct from the NEES‑based detector.
- **Proposed VA** — the three‑layer method above.

All four methods' per‑step wall‑clock time was **measured directly**, not estimated.

---

## 📊 Results

### Headline table

| Method | RMSE Mean (m) | RMSE Std (m) | Mean NEES | Avail. < 3 Anch. |
|--------|---------------|--------------|-----------|------------------|
| Naive | 3.56 | 0.09 | 5.48 | 0.00% |
| Reject | 3.61 | 3.00 | 7.70 | 6.51% |
| k‑NN NLOS | 3.93 | 0.13 | 5.68 | 1.59% |
| **Proposed VA** | **1.20** | **1.08** | **6.90** | **0.00%** |

**Headline result:** 66% mean‑RMSE reduction vs. naive, 67% vs. rejection, with tighter spread than rejection and zero anchor‑availability loss.

**An honest negative result worth keeping visible:** the k‑NN baseline did **not** outperform naive — it both misses genuine NLOS events and false‑flags LOS anchors (degrading geometry more than it corrects bias), and it was, contrary to its "lightweight" framing, the **most computationally expensive** method tested (0.93 ms/step vs. 0.48 ms for the proposed method), due to its $O(\text{window}^2)$ pairwise‑distance computation. This matters for the paper's argument: the gain here comes specifically from coupling the filter's own consistency statistic (NEES) to geometric recovery, not from "having some ML‑flavored detector."

### Instrumented safety‑layer behavior (measured, not assumed)

Of steps with at least one anchor flagged NLOS: the whole‑vector NIS check skips the full update on **7.5%** of them; the per‑row NIS check separately soft‑rejects a further **0.8%** of individual VA measurements that pass the whole‑vector test. Roughly 1 in 13 flagged steps would otherwise have injected an update the safety layer judged inconsistent with the filter's own covariance.

### Computational cost (measured)

| Method | Time / step (ms) |
|--------|------------------|
| Naive | 0.15 |
| Reject | 0.15 |
| k‑NN NLOS | 0.93 |
| **Proposed VA** | **0.48** |

All methods are ~2 orders of magnitude below the 100 ms budget of a 10 Hz update rate.

---

## 🖼️ Figures

All figures are available as vector PDFs in the [`Results/Figures/`](Results/Figures/) directory.

| Figure | Description |
|--------|-------------|
| [`fig_environment.pdf`](Results/Figures/fig_environment.pdf) | Hall layout — real anchors (green), NLOS wall blockers (gray), one representative true trajectory (blue), and every virtual‑anchor location spawned across that trial (red stars). |
| [`fig_rmse_boxplot.pdf`](Results/Figures/fig_rmse_boxplot.pdf) | Per‑trial RMSE distribution across all 50 Monte Carlo trials, all four methods side by side. |
| [`fig_cdf.pdf`](Results/Figures/fig_cdf.pdf) | CDF of per‑trial RMSE, four methods — shows the proposed method's advantage holds across the full trial distribution, not just in the mean. |
| [`fig_nees_timeline.pdf`](Results/Figures/fig_nees_timeline.pdf) | One representative trial: per‑anchor NEES over time with true NLOS episodes shaded and VA‑activation onsets marked (top panel), and position error over time for naive/reject/proposed VA (bottom panel) — shows NEES excursions coinciding with real NLOS episodes and VA activation following shortly after. |
| [`fig_computational.pdf`](Results/Figures/fig_computational.pdf) | Measured per‑step wall‑clock runtime, four methods. |

Raw numeric results backing these figures/tables are in [`Results/mc_results.npz`](Results/mc_results.npz) (NumPy archive: one `(50, 3)` array per method — columns are `[RMSE, mean_NEES, availability_%]` per trial).

---

## 📝 Scope, Honesty Notes, and What's Not Claimed

- This is **simulation‑only**, in a synthetic hall with a synthetic trajectory — not validated on real hardware or a real building yet.
- The novelty claim (analyzing this specific feedback‑loop mechanism in VA literature) is stated as an observation from the related work I've reviewed so far, not a verified, exhaustive priority claim.
- One reference in the paper's bibliography is only partially verified (confirmed content/title, not independently confirmed venue/year against the primary source) — flagged as such in the paper itself rather than presented as fully checked.
- The IMU‑drift robustness discussion in the paper is an **analytic bound** from the simulation's own noise parameters, not a separately measured quantity — the README and paper both say so explicitly rather than blurring the two.

---

## ⚖️ On Priority and Reuse of This Work

I want people — professors, reviewers, other students — to be able to see what I actually did here, in enough depth to evaluate the contribution. I also don't want to pretend a README can do something it can't: **there is no wording that technically prevents someone from reading this page and reimplementing the idea.** That's true of research generally, not a gap specific to this repo, and I'd rather say that plainly than write something reassuring but false.

What actually *does* establish priority, in order of how much it matters:

1. **A dated, public trail of the work.** This repository's commit history, and (once posted) a preprint, are both time‑stamped evidence of when this idea existed in this specific, developed form — with math, working results, and analysis — not just as a one‑line idea. Priority disputes in research are resolved by "who published/timestamped it first and most concretely," not by who kept it secret longest.
2. **A preprint (e.g., arXiv) once the paper is far enough along.** This is the standard mechanism the academic community actually trusts for priority — more than a GitHub timestamp alone.
3. **Standard copyright**, which already applies automatically to this README's text, the figures, and (once released) the code — no registration needed. It protects *this specific expression* of the work (the writing, the exact figures, the exact code), not the underlying method as an abstract idea. That's how copyright works everywhere, not a weakness of this particular repo.
4. **Citation norms.** If you use ideas, figures, or results from this repository or the associated paper, please cite it — a formal citation (BibTeX) will be added here once the paper has a DOI/arXiv ID.

If you're evaluating this project (e.g., as a professor or reviewer) and want early access to the implementation ahead of the code release, please reach out directly rather than asking for it to be posted early — contact details below.

---

## 📬 Contact

**Muhammad Ibaad**  
Dawood University of Engineering and Technology, Karachi, Pakistan  
ibaadsajidshaikh18@gmail.com  

*A preprint/paper for this work is in progress. Citation details will be added here once available.*

---

<p align="center">
  <sub>Built with ❤️ using MATLAB & Python · Last updated: 2026-07-12</sub>
</p>
