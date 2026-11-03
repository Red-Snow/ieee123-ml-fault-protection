"""Load the simulated dataset and build noisy features + labels."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from features import add_noise, build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["SLG", "LL", "LLG", "LLL"]
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]


def load(path: str = os.path.join(ROOT, "data", "raw_phasors.npz"), seed: int = 0):
    raw = np.load(path)
    meta = pd.read_csv(path.replace(".npz", "_meta.csv"))
    rng = np.random.default_rng(seed)
    ph = {k: add_noise(raw[k], rng) for k in ("v_pre", "v_post", "i_pre", "i_post")}
    X = build(ph["v_pre"], ph["v_post"], ph["i_pre"], ph["i_post"])
    meta["is_fault"] = (meta.ftype != "NORMAL").astype(int)
    return X, meta, ph
