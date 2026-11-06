"""Turn raw phasors into the feature vector a protection device would see.

Per voltage sensor (bus):   |V| a,b,c (pu), angle deviation a,b,c (deg),
                            |V0|/|V1|, |V2|/|V1|                       -> 8
Per current sensor (line):  |I| a,b,c (kA), angle deviation a,b,c (deg),
                            delta |I| a,b,c vs. pre-event (kA),
                            |I0|, |I2| (kA)                           -> 11

15 x 8 + 8 x 11 = 208 features. Measurement noise is added to the phasors *before*
features are computed (see `add_noise`).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import I_SENSORS, NOISE_ANG_SD_DEG, NOISE_MAG_SD, NOMINAL_ANG, V_SENSORS

A = np.exp(2j * np.pi / 3)
SEQ = np.array([[1, 1, 1], [1, A, A ** 2], [1, A ** 2, A]]) / 3  # rows: zero, positive, negative
V_FEATS = ["ma", "mb", "mc", "aa", "ab", "ac", "v0r", "v2r"]
I_FEATS = ["ma", "mb", "mc", "aa", "ab", "ac", "dma", "dmb", "dmc", "i0", "i2"]
PHASOR_KEYS = ("v_pre", "v_post", "i_pre", "i_post")


def add_noise(z: np.ndarray, rng: np.random.Generator,
              mag_sd: float = NOISE_MAG_SD, ang_sd_deg: float = NOISE_ANG_SD_DEG) -> np.ndarray:
    """Multiplicative magnitude noise and additive angle noise (1 sigma)."""
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
    """Arrays shaped (n, sensors, 3) complex -> features (n, 208) float32."""
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


def build_from(ph: dict[str, np.ndarray]) -> np.ndarray:
    """`build` for a phasor dict with keys v_pre, v_post, i_pre, i_post."""
    return build(*(ph[k] for k in PHASOR_KEYS))


def feature_names() -> list[str]:
    return ([f"V{b}_{q}" for b in V_SENSORS for q in V_FEATS]
            + [f"I{line}_{q}" for line in I_SENSORS for q in I_FEATS])


@lru_cache(maxsize=1)
def _sensor_columns() -> tuple[tuple[str, tuple[int, ...]], ...]:
    out, k = [], 0
    for b in V_SENSORS:
        out.append((f"V{b}", tuple(range(k, k + len(V_FEATS)))))
        k += len(V_FEATS)
    for line in I_SENSORS:
        out.append((f"I{line}", tuple(range(k, k + len(I_FEATS)))))
        k += len(I_FEATS)
    return tuple(out)


def sensor_columns() -> dict[str, list[int]]:
    """Which feature columns belong to each sensor, e.g. {'V149': [0..7], ..., 'IL67': [...]}."""
    return {s: list(c) for s, c in _sensor_columns()}
