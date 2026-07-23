"""Monte Carlo: 50 trials x {naive, reject, VA}. Metrics: RMSE, NEES consistency, availability."""
import numpy as np
from env import ANCHORS, WALLS
from trajectory import generate_trajectory, DT, N
from sensors import generate_rssi, generate_imu
from ekf_nees import EKF_NEES
from ekf_va import EKF_VA
from knn_detector import KnnNlosDetector
from ekf import EKF
from sensors import rssi_model

N_TRIALS = 50


def run_trial_knn(seed):
    """k-NN NLOS baseline: detect via online LOF-style outlier test on RSSI residuals,
    reject (R=1e6) flagged anchors -- same rejection mechanism as 'reject' mode, but
    with a different (lightweight ML) detector in place of NEES."""
    rng = np.random.default_rng(seed)
    traj = generate_trajectory(seed=seed)
    x0 = np.array([traj["px_true"][0], traj["py_true"][0], traj["vx"][0], traj["vy"][0], 0.0, 0.0])
    ekf = EKF(x0)
    det = KnnNlosDetector(n_anchors=6)
    err2 = np.zeros(N)
    eps_all = []
    n_usable = np.zeros(N)
    bias_state = np.zeros(2)
    for k in range(N):
        if k > 0:
            ekf.predict(DT)
        tag_true = np.array([traj["px_true"][k], traj["py_true"][k]])
        rssi, _ = generate_rssi(ANCHORS, tag_true, WALLS, rng)
        vel_prev = np.array([traj["vx"][max(k - 1, 0)], traj["vy"][max(k - 1, 0)]])
        vel_curr = np.array([traj["vx"][k], traj["vy"][k]])
        acc, bias_state = generate_imu(vel_prev, vel_curr, DT, bias_state, rng)

        pos_est = ekf.x[:2]
        residuals = np.array([rssi[i] - rssi_model(ANCHORS[i], pos_est) for i in range(6)])
        flags = det.update(residuals)

        sigma_nom = np.full(6, 2.0)
        H = np.zeros((8, 6))
        for i, a in enumerate(ANCHORS):
            from ekf import H_rssi_row
            H[i, :] = H_rssi_row(ekf.x, a)
        H[6, 4] = 1.0
        H[7, 5] = 1.0
        z_hat = ekf.predicted_z(ANCHORS)
        z = np.concatenate([rssi, acc])
        y = z - z_hat
        R = np.concatenate([sigma_nom, [0.05, 0.05]]) ** 2
        S = H @ ekf.P @ H.T + np.diag(R)
        eps = (y[:6] ** 2) / np.maximum(np.diag(S)[:6], 1e-6)

        sigma_eff = sigma_nom.copy()
        sigma_eff[flags] = np.sqrt(1e6)
        ekf.update(z, ANCHORS, sigma_eff)

        err2[k] = (ekf.x[0] - tag_true[0]) ** 2 + (ekf.x[1] - tag_true[1]) ** 2
        eps_all.append(eps)
        n_usable[k] = 6 - flags.sum()

    rmse = np.sqrt(np.mean(err2))
    mean_nees = np.mean(np.concatenate(eps_all))
    avail = np.mean(n_usable < 3) * 100.0
    return rmse, mean_nees, avail


def _init_ekf(cls, traj):
    x0 = np.array([traj["px_true"][0], traj["py_true"][0], traj["vx"][0], traj["vy"][0], 0.0, 0.0])
    return cls(x0)


def run_trial_naive_reject(seed, mode):
    rng = np.random.default_rng(seed)
    traj = generate_trajectory(seed=seed)
    ekf = _init_ekf(EKF_NEES, traj)
    err2 = np.zeros(N)
    eps_all = []
    n_usable = np.zeros(N)
    bias_state = np.zeros(2)
    for k in range(N):
        tag_true = np.array([traj["px_true"][k], traj["py_true"][k]])
        rssi, _ = generate_rssi(ANCHORS, tag_true, WALLS, rng)
        vel_prev = np.array([traj["vx"][max(k - 1, 0)], traj["vy"][max(k - 1, 0)]])
        vel_curr = np.array([traj["vx"][k], traj["vy"][k]])
        acc, bias_state = generate_imu(vel_prev, vel_curr, DT, bias_state, rng)
        z = np.concatenate([rssi, acc])
        flags, eps = ekf.step(z, ANCHORS, DT, mode=mode)
        err2[k] = (ekf.x[0] - tag_true[0]) ** 2 + (ekf.x[1] - tag_true[1]) ** 2
        eps_all.append(eps)
        n_usable[k] = 6 if mode == "naive" else (6 - flags.sum())
    rmse = np.sqrt(np.mean(err2))
    mean_nees = np.mean(np.concatenate(eps_all))
    avail = np.mean(n_usable < 3) * 100.0
    return rmse, mean_nees, avail


def run_trial_va(seed):
    rng = np.random.default_rng(seed)
    traj = generate_trajectory(seed=seed)
    ekf = _init_ekf(EKF_VA, traj)
    err2 = np.zeros(N)
    eps_all = []
    n_usable = np.zeros(N)
    bias_state = np.zeros(2)
    for k in range(N):
        tag_true = np.array([traj["px_true"][k], traj["py_true"][k]])
        rssi, _ = generate_rssi(ANCHORS, tag_true, WALLS, rng)
        vel_prev = np.array([traj["vx"][max(k - 1, 0)], traj["vy"][max(k - 1, 0)]])
        vel_curr = np.array([traj["vx"][k], traj["vy"][k]])
        acc, bias_state = generate_imu(vel_prev, vel_curr, DT, bias_state, rng)
        flags, eps, _ = ekf.step(rssi, acc, ANCHORS, rng, DT)
        err2[k] = (ekf.x[0] - tag_true[0]) ** 2 + (ekf.x[1] - tag_true[1]) ** 2
        eps_all.append(eps)
        n_usable[k] = 6  # VA substitutes rather than dropping -> always 6 usable
    rmse = np.sqrt(np.mean(err2))
    mean_nees = np.mean(np.concatenate(eps_all))
    avail = np.mean(n_usable < 3) * 100.0
    return rmse, mean_nees, avail


def run_monte_carlo(n_trials=N_TRIALS):
    results = {"naive": [], "reject": [], "knn": [], "va": []}
    for trial in range(n_trials):
        seed = 1000 + trial
        results["naive"].append(run_trial_naive_reject(seed, mode="naive"))
        results["reject"].append(run_trial_naive_reject(seed, mode="reject"))
        results["knn"].append(run_trial_knn(seed))
        results["va"].append(run_trial_va(seed))
        if (trial + 1) % 10 == 0:
            print(f"  trial {trial + 1}/{n_trials} done")
    for k in results:
        results[k] = np.array(results[k])  # (n_trials, 3) cols: rmse, nees, avail
    return results


if __name__ == "__main__":
    res = run_monte_carlo()
    np.savez("mc_results.npz", **res)
    for method in ["naive", "reject", "va"]:
        r = res[method]
        print(f"{method}: RMSE mean={r[:,0].mean():.3f} std={r[:,0].std():.3f} | "
              f"NEES mean={r[:,1].mean():.2f} | avail(<3anch)={r[:,2].mean():.2f}%")
