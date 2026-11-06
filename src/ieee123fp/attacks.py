"""False-data injection (FDI) by replay.

Threat model: an attacker controls the data streams of k of the 23 measurement devices and
replays real recorded data:
  * masking  - during a real fault, compromised sensors report their pre-fault readings
               -> goal: hide the fault (no trip)
  * spoofing - during normal operation, compromised sensors replay readings recorded
               during a real fault -> goal: cause a false trip
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import I_SENSORS, SENSORS, V_SENSORS


def _locate(sensor: str) -> tuple[str, int]:
    """'V149' -> ('v', 0); 'ISw2' -> ('i', 1)."""
    if sensor[0] == "V":
        return "v", V_SENSORS.index(sensor[1:])
    return "i", I_SENSORS.index(sensor[1:])


def replay_attack(ph: dict, idx: np.ndarray, sensors: Sequence[Sequence[str]], mode: str,
                  donor_ph: dict | None = None, donor_idx: np.ndarray | None = None) -> dict:
    """Attacked copies of the phasor dict for rows `idx`.

    sensors[r] lists the compromised sensors for row r. mode="mask" replays each sensor's own
    pre-event reading; mode="spoof" replays donor_ph[...][donor_idx[r]] (a recorded fault)."""
    if mode not in ("mask", "spoof"):
        raise ValueError(mode)
    a = {k: v[idx].copy() for k, v in ph.items()}
    for r, sens in enumerate(sensors):
        for s in sens:
            kind, j = _locate(s)
            if mode == "mask":
                a[f"{kind}_post"][r, j] = a[f"{kind}_pre"][r, j]
            else:
                d = donor_idx[r]
                a[f"{kind}_post"][r, j] = donor_ph[f"{kind}_post"][d, j]
                a[f"{kind}_pre"][r, j] = donor_ph[f"{kind}_pre"][d, j]
    return a


def choose_sensors(attacker: str, k: int, n: int, rng: np.random.Generator,
                   ranked: Sequence[str]) -> list[list[str]]:
    """Compromised sensor sets for n events: 'random' draws k per event; 'targeted' always
    takes the k sensors the detector relies on most (`ranked`, most important first)."""
    if attacker == "random":
        return [list(rng.choice(SENSORS, k, replace=False)) for _ in range(n)]
    if attacker == "targeted":
        return [list(ranked[:k]) for _ in range(n)]
    raise ValueError(attacker)
