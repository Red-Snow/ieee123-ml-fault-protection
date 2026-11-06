"""Defences against false-data injection.

ResilientVote (works): f-resilient voting, in the spirit of the W-MSR rule used for
cyber-secure distributed microgrid control. Each sensor gets its OWN small fault detector
that only sees its own readings, set like a relay to fire on at most `fpr` of normal events.
The breaker trips only if at least f+1 sensors independently say "fault". An attacker who
controls at most f sensors can then never cause a trip on their own, and can hide a fault
only if fewer than 2f+1 honest sensors saw it. f trades security against sensitivity to
faults that only a few sensors can see.

ConsistencyCheck (did NOT work, kept for the record): for every sensor, a regressor predicts
its readings from all OTHER sensors; inconsistent sensors are flagged and their readings are
replaced by the prediction. It fails because two or more compromised sensors back each other
up, and it flags ~8 % of clean events.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier

from ..config import SENSORS
from ..features import sensor_columns

log = logging.getLogger(__name__)


class ResilientVote:
    def __init__(self, fpr: float = 0.01, max_iter: int = 200, random_state: int = 0):
        self.fpr, self.max_iter, self.random_state = fpr, max_iter, random_state
        self.cols = sensor_columns()

    def fit(self, X, y, X_cal, y_cal) -> ResilientVote:
        """One detector per sensor; each threshold is set on calibration data so the sensor
        fires on at most `fpr` of normal (non-fault) events (and never below p = 0.5)."""
        self.m, self.thr = {}, {}
        for s in SENSORS:
            c = self.cols[s]
            self.m[s] = HistGradientBoostingClassifier(max_iter=self.max_iter, random_state=self.random_state).fit(X[:, c], y)
            p = self.m[s].predict_proba(X_cal[y_cal == 0][:, c])[:, 1]
            self.thr[s] = max(np.quantile(p, 1 - self.fpr), 0.5)
        return self

    def sensor_votes(self, X) -> np.ndarray:
        """(n, 23) bool: which sensors say 'fault'."""
        return np.stack([self.m[s].predict_proba(X[:, self.cols[s]])[:, 1] > self.thr[s] for s in SENSORS], 1)

    def votes(self, X) -> np.ndarray:
        """(n,) number of sensors that say 'fault'."""
        return self.sensor_votes(X).sum(1)

    def predict(self, X, f: int) -> np.ndarray:
        """Trip (1) only if more than f sensors independently see a fault."""
        return (self.votes(X) >= f + 1).astype(int)


class ConsistencyCheck:
    def __init__(self, quantile: float = 0.995):
        self.quantile = quantile
        self.cols = sensor_columns()

    def fit(self, X, X_cal) -> ConsistencyCheck:
        self.models, self.sd, self.thr = {}, {}, {}
        for s in SENSORS:
            c = self.cols[s]
            other = np.setdiff1d(np.arange(X.shape[1]), c)
            m = ExtraTreesRegressor(n_estimators=40, max_features=0.3, min_samples_leaf=2, n_jobs=-1, random_state=0)
            m.fit(X[:, other], X[:, c])
            res = X_cal[:, c] - m.predict(X_cal[:, other])
            sd = res.std(0) + 1e-6
            score = np.sqrt(((res / sd) ** 2).mean(1))
            self.models[s], self.sd[s], self.thr[s] = (m, other), sd, np.quantile(score, self.quantile)
            log.debug("consistency model %-6s thr=%.2f", s, self.thr[s])
        return self

    def check_and_repair(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Flag inconsistent sensors; replace their features by the prediction from the rest."""
        flags = np.zeros((len(X), len(SENSORS)), bool)
        preds = {}
        for j, s in enumerate(SENSORS):
            m, other = self.models[s]
            p = m.predict(X[:, other])
            score = np.sqrt((((X[:, self.cols[s]] - p) / self.sd[s]) ** 2).mean(1))
            flags[:, j] = score > self.thr[s]
            preds[s] = p
        Xr = X.copy()
        for j, s in enumerate(SENSORS):
            rows = np.where(flags[:, j])[0]
            Xr[np.ix_(rows, self.cols[s])] = preds[s][rows]
        return flags, Xr
