"""RSSI + IMU sensor models."""
import numpy as np
from los_checker import los_mask

PL_D0 = -40.0  # dBm at 1m ref
N_PLE = 2.5
D0 = 1.0
SIGMA_LOS = 2.0
NLOS_EXCESS_MEAN = 8.0
NLOS_EXCESS_STD = 4.0
SIGMA_IMU = 0.05  # m/s^2


def rssi_model(anchor_pos, tag_pos):
    d = max(np.linalg.norm(np.asarray(anchor_pos) - np.asarray(tag_pos)), 0.1)
    return PL_D0 - 10 * N_PLE * np.log10(d / D0)


def generate_rssi(anchors, tag_pos, walls, rng):
    mask = los_mask(anchors, tag_pos, walls)  # True = LOS
    rssi = np.zeros(len(anchors))
    for i, a in enumerate(anchors):
        base = rssi_model(a, tag_pos)
        noise = rng.normal(0, SIGMA_LOS)
        if not mask[i]:
            noise += rng.normal(NLOS_EXCESS_MEAN, NLOS_EXCESS_STD)
        rssi[i] = base + noise
    return rssi, mask


def generate_imu(tag_vel_prev, tag_vel_curr, dt, bias_state, rng):
    """Return noisy accel (ax, ay), update bias random walk."""
    true_acc = (np.asarray(tag_vel_curr) - np.asarray(tag_vel_prev)) / dt
    bias_state = bias_state + rng.normal(0, 0.005, 2)  # slow bias walk
    noisy_acc = true_acc + bias_state + rng.normal(0, SIGMA_IMU, 2)
    return noisy_acc, bias_state
