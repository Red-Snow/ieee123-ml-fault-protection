"""Topology-aware (physics-informed) fault-zone localisation.

Why: a black-box classifier maps "this pattern of measurements" -> "zone". When the grid is
reconfigured, the same zone produces a different pattern (fault current now arrives through
a different switch), so the mapping breaks.

The idea, in two steps:
  1. A per-sensor ML model answers a LOCAL question that means the same thing in every
     layout: "is fault current flowing through me, and which way?" (forward / backward /
     not at all). Its only inputs are the sensor's own incremental current projected onto
     the substation's incremental current, and the size of its change relative to the
     largest change on the feeder.
  2. The current switch states -- which the utility's SCADA already knows -- give the grid
     graph. For every bus we know which sensors sit on the path(s) from the substation to
     that bus, and in which direction (`grid.path_signatures`). We pick the zone whose buses
     best explain the per-sensor answers (a likelihood over buses, summed per zone).

The per-sensor model never needs to see the layout it is used on; only the graph does.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import I_SENSORS, SUBSTATION_SENSOR, ZONES
from ..features import sensor_columns
from ..grid import compute_zones, path_signatures

N_LOCAL = 4  # local features per current sensor


def local_features(X: np.ndarray, ph: dict, flip: bool = False) -> np.ndarray:
    """(n, 208) features + phasors -> (n, n_current_sensors, 4) layout-independent features:
    Re/Im/|.| of the sensor's incremental current projected on the substation's, and its
    largest phase change relative to the largest change on the feeder.

    No sensor identity is included, so every sensor shares one "which way is the fault
    current flowing?" rule. flip=True gives the same event as seen by a sensor wired
    backwards: every current phasor rotates by 180 deg, so the projection changes sign and
    the direction label flips. This exact physical symmetry is used as data augmentation."""
    cols = sensor_columns()
    blocks = [X[:, cols[f"I{s}"]] for s in I_SENSORS]          # each (n, 11)
    dmax = [np.abs(b[:, 6:9]).max(1) for b in blocks]           # largest |delta I| over phases
    big = np.maximum(np.max(np.stack(dmax, 1), 1), 1e-6)
    dI = (ph["i_post"] - ph["i_pre"]) / 1000.0                  # incremental (superimposed) current, kA
    # Direction is judged against the substation breaker's own incremental current -- the
    # same polarising idea a directional relay uses, which removes the dependence on fault
    # resistance and type. Weighting by the reference itself makes phases the fault does
    # not touch contribute ~0:
    #   proj_k = sum_p dI_k,p * conj(ref_p) / sum_p |ref_p|^2
    # Re > 0: same direction as at the substation (forward); Re < 0: backwards; ~0: off path.
    ref = dI[:, I_SENSORS.index(SUBSTATION_SENSOR), :]
    proj = (dI * np.conj(ref)[:, None, :]).sum(2) / np.maximum((np.abs(ref) ** 2).sum(1), 1e-9)[:, None]
    if flip:
        proj = -proj
    # Only physically meaningful quantities are kept. Raw magnitudes/angles let the model
    # memorise layout-specific patterns (loop accuracy collapsed); see README.
    return np.stack([np.stack([proj[:, k].real, proj[:, k].imag, np.abs(proj[:, k]), dmax[k] / big], 1)
                     for k in range(len(I_SENSORS))], 1)


class TopologyAwareZoneLocator:
    """Per-sensor fault-current direction model + graph decoder.

    >>> loc = TopologyAwareZoneLocator().fit(X, ph, topos, buses)
    >>> loc.predict(X_new, ph_new, topo="T3_loop")  # zones, using that layout's graph
    """

    def __init__(self, max_iter: int = 300, random_state: int = 0,
                 division_range: tuple[float, float] = (0.05, 1.0)):
        self.max_iter = max_iter
        self.random_state = random_state
        self.division_range = division_range
        self.zones = compute_zones()

    # ------------------------------------------------------------------
    @staticmethod
    def direction_labels(topos, buses) -> np.ndarray:
        """(n, n_current_sensors) in {-1, 0, +1}: the true direction at each sensor."""
        sigs = {t: path_signatures(t) for t in set(topos)}
        return np.stack([sigs[t][str(b).lower()] for t, b in zip(topos, buses)])

    def fit(self, X: np.ndarray, ph: dict, topos, buses) -> TopologyAwareZoneLocator:
        """Train on fault events from any layouts (topos[i], buses[i] = layout and faulted bus)."""
        y = self.direction_labels(topos, buses).reshape(-1)
        R = local_features(X, ph).reshape(-1, N_LOCAL)
        R_flip = local_features(X, ph, flip=True).reshape(-1, N_LOCAL)
        # Augmentation 1 (direction symmetry): the same event seen by a reversed sensor.
        # Augmentation 2 (current division): in a meshed grid the fault current splits over
        # parallel paths, so an on-path sensor may carry only part of it. Scale on-path rows
        # by a random factor; labels unchanged. Both facts never appear in radial layouts.
        rng = np.random.default_rng(self.random_state)
        Rd = R.copy()
        k = rng.uniform(*self.division_range, len(Rd))
        on = y != 0
        Rd[on] *= k[on, None]
        self.clf_ = HistGradientBoostingClassifier(max_iter=self.max_iter, random_state=self.random_state).fit(
            np.vstack([R, R_flip, Rd, -Rd * np.array([1, 1, -1, -1])]),
            np.concatenate([y, -y, y, -y]))
        self.classes_ = np.array(self.clf_.classes_)                # [-1, 0, 1]
        return self

    # ------------------------------------------------------------------
    def direction_proba(self, X: np.ndarray, ph: dict) -> np.ndarray:
        """(n, n_current_sensors, 3) probabilities over classes_ (-1, 0, +1)."""
        R = local_features(X, ph).reshape(-1, N_LOCAL)
        return self.clf_.predict_proba(R).reshape(len(X), len(I_SENSORS), len(self.classes_))

    def predict_directions(self, X: np.ndarray, ph: dict) -> np.ndarray:
        return self.classes_[self.direction_proba(X, ph).argmax(2)]

    def bus_log_likelihood(self, X: np.ndarray, ph: dict, topo: str) -> tuple[list[str], np.ndarray]:
        """Log-likelihood of a fault at each bus of layout `topo` given the sensor answers."""
        sig = path_signatures(topo)
        bus_list = list(sig)
        B = np.stack([sig[b] for b in bus_list])                       # (n_bus, S)
        logP = np.log(self.direction_proba(X, ph).clip(1e-4, 1))
        ll = sum(logP[:, :, i] @ (B == c).T.astype(float) for i, c in enumerate(self.classes_))
        return bus_list, ll                                            # (n, n_bus)

    def predict_proba(self, X: np.ndarray, ph: dict, topo: str) -> np.ndarray:
        """(n, 5) zone posterior (uniform prior over buses)."""
        bus_list, ll = self.bus_log_likelihood(X, ph, topo)
        zb = np.array([self.zones[b] for b in bus_list])
        p = np.exp(ll - ll.max(1, keepdims=True))
        post = np.stack([p[:, zb == z].sum(1) for z in ZONES], 1)
        return post / post.sum(1, keepdims=True)

    def predict(self, X: np.ndarray, ph: dict, topo: str) -> np.ndarray:
        return np.array(ZONES)[self.predict_proba(X, ph, topo).argmax(1)]
