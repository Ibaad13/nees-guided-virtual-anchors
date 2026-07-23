"""VA placement via grid-search GDOP minimization."""
import numpy as np
from env import HALL_W, HALL_H

GRID_RES = 0.5
MIN_VA_STANDOFF = 2.0  # m; avoid near-field log-distance hypersensitivity


def _grid_points():
    xs = np.arange(0, HALL_W + 1e-9, GRID_RES)
    ys = np.arange(0, HALL_H + 1e-9, GRID_RES)
    XX, YY = np.meshgrid(xs, ys)
    return np.stack([XX.ravel(), YY.ravel()], axis=1)  # (600ish, 2)


GRID = _grid_points()


def gdop(anchor_positions, tag_pos):
    """anchor_positions: (M,2) list incl. candidate. tag_pos: (2,) current estimate."""
    H = np.zeros((len(anchor_positions), 2))
    for i, a in enumerate(anchor_positions):
        d = max(np.linalg.norm(tag_pos - a), 0.1)
        H[i, 0] = -(tag_pos[0] - a[0]) / d
        H[i, 1] = -(tag_pos[1] - a[1]) / d
    try:
        Ginv = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return np.inf
    tr = np.trace(Ginv)
    if tr < 0:
        return np.inf
    return np.sqrt(tr)


def optimize_va(los_anchor_positions, tag_est, cache=None, cache_key=None, tol_gdop=0.05):
    """Grid search best VA position minimizing GDOP(LOS anchors + candidate).
    cache: dict {anchor_idx: (pos, gdop_val)} for reuse if still near-optimal.
    Returns va_pos (2,)."""
    base = np.array(los_anchor_positions)
    # fixed part A = H_base^T H_base (2x2), vectorized analytic 2x2 GDOP over grid
    Hb = np.zeros((len(base), 2))
    for i, a in enumerate(base):
        d = max(np.linalg.norm(tag_est - a), 0.1)
        Hb[i, 0] = -(tag_est[0] - a[0]) / d
        Hb[i, 1] = -(tag_est[1] - a[1]) / d
    A = Hb.T @ Hb  # 2x2

    dxy = tag_est[None, :] - GRID  # (Ngrid, 2)
    dist = np.maximum(np.linalg.norm(dxy, axis=1), 0.1)
    r = -dxy / dist[:, None]  # (Ngrid, 2) candidate rows

    a11 = A[0, 0] + r[:, 0] ** 2
    a22 = A[1, 1] + r[:, 1] ** 2
    a12 = A[0, 1] + r[:, 0] * r[:, 1]
    det = a11 * a22 - a12 ** 2
    det = np.where(det > 1e-9, det, np.inf)
    trace_inv = (a11 + a22) / det
    gdop_vals = np.sqrt(np.maximum(trace_inv, 0))
    gdop_vals = np.where(dist >= MIN_VA_STANDOFF, gdop_vals, np.inf)  # exclude near-field candidates
    best_gdop = gdop_vals.min()

    if cache is not None and cache_key in cache:
        prev_pos, _ = cache[cache_key]
        if np.linalg.norm(tag_est - prev_pos) >= MIN_VA_STANDOFF:
            prev_g = gdop(np.vstack([base, prev_pos]), tag_est)
            if prev_g <= best_gdop + tol_gdop:
                return prev_pos, prev_g

    # near-optimal set -> pick closest to tag_est
    near_idx = np.where(gdop_vals <= best_gdop + 1e-6)[0]
    if len(near_idx) > 1:
        dists = np.linalg.norm(GRID[near_idx] - tag_est, axis=1)
        best_pos = GRID[near_idx[int(np.argmin(dists))]]
    else:
        best_pos = GRID[near_idx[0]]

    if cache is not None:
        cache[cache_key] = (best_pos, best_gdop)
    return best_pos, best_gdop
