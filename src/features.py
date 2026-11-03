"""
Turn raw phasors into the feature vector a protection device would see.

Per voltage sensor (bus):   |V| a,b,c (pu), angle deviation a,b,c (deg),
                            |V0|/|V1|, |V2|/|V1|                       -> 8
Per current sensor (line):  |I| a,b,c (kA), angle deviation a,b,c (deg),
                            delta |I| a,b,c vs. pre-event (kA),
                            |I0|, |I2| (kA)                           -> 11

Measurement noise (instrument-transformer + PMU error) is added to the phasors
*before* features are computed: 0.5 % magnitude (1 sigma), 0.2 deg angle.
"""
from __future__ import annotations

import numpy as np

from sim import I_SENSORS, NOMINAL_ANG, V_SENSORS

A = np.exp(2j * np.pi / 3)
SEQ = np.array([[1, 1, 1], [1, A, A ** 2], [1, A ** 2, A]]) / 3  # rows: zero, positive, negative


def add_noise(z: np.ndarray, rng: np.random.Generator, mag_sd=0.005, ang_sd_deg=0.2) -> np.ndarray:
    mag = np.abs(z) * (1 + rng.normal(0, mag_sd, z.shape))
    ang = np.angle(z) + np.deg2rad(rng.normal(0, ang_sd_deg, z.shape))
    return mag * np.exp(1j * ang)


def _ang_dev(z: np.ndarray) -> np.ndarray:
    """Angle relative to nominal phase angle, wrapped to [-180, 180]; 0 where phase missing."""
    d = np.angle(z, deg=True) - NOMINAL_ANG
    d = (d + 180) % 360 - 180
    return np.where(np.abs(z) > 1e-6, d, 0.0)


def _seq(z: np.ndarray) -> np.ndarray:
    return np.abs(np.einsum("sp,...p->...s", SEQ, z))  # (..., 3): |X0|, |X1|, |X2|


def build(v_pre, v_post, i_pre, i_post) -> np.ndarray:
    """Arrays shaped (n, sensors, 3) complex -> features (n, n_features)."""
    n = v_post.shape[0]
    vm = np.abs(v_post)
    vs = _seq(v_post)
    v1 = np.maximum(vs[..., 1], 1e-3)
    vf = np.concatenate([vm, _ang_dev(v_post), (vs[..., 0] / v1)[..., None], (vs[..., 2] / v1)[..., None]], axis=-1)

    ip, iq = i_post / 1000.0, i_pre / 1000.0  # kA
    im = np.abs(ip)
    is_ = _seq(ip)
    # current angle relative to nominal too (power-factor angle + any shift)
    i_f = np.concatenate([im, _ang_dev(ip), im - np.abs(iq), is_[..., [0]], is_[..., [2]]], axis=-1)
    return np.concatenate([vf.reshape(n, -1), i_f.reshape(n, -1)], axis=1).astype(np.float32)


def feature_names() -> list[str]:
    names = []
    for b in V_SENSORS:
        names += [f"V{b}_{q}" for q in ["ma", "mb", "mc", "aa", "ab", "ac", "v0r", "v2r"]]
    for l in I_SENSORS:
        names += [f"I{l}_{q}" for q in ["ma", "mb", "mc", "aa", "ab", "ac", "dma", "dmb", "dmc", "i0", "i2"]]
    return names


# which feature columns belong to each sensor (used by the attack + defence code)
def sensor_columns() -> dict[str, list[int]]:
    cols, k = {}, 0
    for b in V_SENSORS:
        cols[f"V{b}"] = list(range(k, k + 8)); k += 8
    for l in I_SENSORS:
        cols[f"I{l}"] = list(range(k, k + 11)); k += 11
    return cols
