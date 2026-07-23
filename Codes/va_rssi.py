"""Generate VA pseudo-RSSI: model-based prediction from VA pos to a SMOOTHED tag estimate.
Uses higher noise (SIGMA_VA) than real LOS anchors since it's a synthetic, model-based
measurement, not a physical one -- prevents it from dominating the update."""
import numpy as np
from sensors import rssi_model

SIGMA_VA = 4.0  # dB, weaker trust than real LOS (SIGMA_LOS=2.0)


def generate_va_rssi(va_pos, tag_est, rng):
    base = rssi_model(va_pos, tag_est)
    return base + rng.normal(0, SIGMA_VA)

