"""
False-data-injection (FDI) robustness of the ML fault detector, and a
cross-sensor consistency check as a defence.

Threat model: an attacker controls the data stream of k measurement devices
(of 23) and replays recorded data:
  * masking  - during a real fault, the compromised sensors report their
               pre-fault (normal) readings  -> goal: hide the fault (no trip)
  * spoofing - during normal operation, the compromised sensors replay readings
               recorded during a real fault -> goal: cause a false trip
Attacker types: "random" picks k sensors at random; "targeted" picks the k
sensors the detector relies on most.

Defence 1 (did NOT work, kept for the record): for every sensor, a regressor
predicts its readings from all OTHER sensors; inconsistent sensors are flagged
and their readings replaced by the prediction. Fails because two or more
compromised sensors back each other up, and it raises ~9 % false alarms.

Defence 2 (works): f-resilient voting, in the spirit of the W-MSR rule used
for cyber-secure distributed microgrid control. Each sensor gets its OWN small
fault detector that only sees its own readings. The breaker trips only if at
least f+1 sensors independently say "fault". An attacker who controls at most
f sensors can then never cause a trip on their own, and can hide a fault only
if fewer than 2f+1 honest sensors saw it. f trades security against
sensitivity to faults that only a few sensors can see.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from data import ROOT, load  # noqa: E402
from features import build, sensor_columns  # noqa: E402
from sim import I_SENSORS, V_SENSORS  # noqa: E402

OUT = os.path.join(ROOT, "results")
SENSORS = [f"V{b}" for b in V_SENSORS] + [f"I{l}" for l in I_SENSORS]
KS = [1, 2, 3, 4, 6, 8, 12]


def attack(ph, idx, sensors, mode, donor_ph=None, donor_idx=None):
    """Return attacked copies of the phasor dict for rows idx."""
    a = {k: v[idx].copy() for k, v in ph.items()}
    for r, sens in enumerate(sensors):
        for s in sens:
            kind, j = (("v", V_SENSORS.index(s[1:])) if s[0] == "V" else ("i", I_SENSORS.index(s[1:])))
            if mode == "mask":
                a[f"{kind}_post"][r, j] = a[f"{kind}_pre"][r, j]
            else:  # spoof: replay a recorded fault at this sensor
                d = donor_idx[r]
                a[f"{kind}_post"][r, j] = donor_ph[f"{kind}_post"][d, j]
                a[f"{kind}_pre"][r, j] = donor_ph[f"{kind}_pre"][d, j]
    return a


def feats(a):
    return build(a["v_pre"], a["v_post"], a["i_pre"], a["i_post"])


class ConsistencyCheck:
    def __init__(self):
        self.cols = sensor_columns()

    def fit(self, X, X_cal):
        self.models, self.sd, self.thr = {}, {}, {}
        for s in SENSORS:
            c = self.cols[s]
            other = np.setdiff1d(np.arange(X.shape[1]), c)
            m = ExtraTreesRegressor(n_estimators=40, max_features=0.3, min_samples_leaf=2, n_jobs=-1, random_state=0)
            m.fit(X[:, other], X[:, c])
            res = X_cal[:, c] - m.predict(X_cal[:, other])
            sd = res.std(0) + 1e-6
            score = np.sqrt(((res / sd) ** 2).mean(1))
            self.models[s], self.sd[s], self.thr[s] = (m, other), sd, np.quantile(score, 0.995)
            print(f"  consistency model {s:6s} thr={self.thr[s]:.2f}", flush=True)
        return self

    def check_and_repair(self, X):
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


class ResilientVote:
    def __init__(self):
        self.cols = sensor_columns()

    def fit(self, X, y, X_cal, y_cal, fpr=0.01):
        """Each sensor's detector is set like a relay: its threshold is chosen so it
        fires on at most `fpr` of normal (non-fault) events in calibration data."""
        self.m, self.thr = {}, {}
        for s in SENSORS:
            c = self.cols[s]
            self.m[s] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[:, c], y)
            p = self.m[s].predict_proba(X_cal[y_cal == 0][:, c])[:, 1]
            self.thr[s] = max(np.quantile(p, 1 - fpr), 0.5)
        return self

    def votes(self, X):
        return np.stack([self.m[s].predict_proba(X[:, self.cols[s]])[:, 1] > self.thr[s] for s in SENSORS], 1).sum(1)


FS = [0, 1, 2, 3, 4, 6]


def run():
    X, meta, ph = load()
    y = meta.is_fault.values
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=meta.ftype, random_state=0)
    idx_fit, idx_cal = train_test_split(idx_tr, test_size=0.2, stratify=meta.ftype.values[idx_tr], random_state=1)

    det = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[idx_tr], y[idx_tr])
    # sensor importance for the targeted attacker (attacker is assumed to know the model)
    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0).fit(X[idx_tr], y[idx_tr])
    cols = sensor_columns()
    imp = {s: rf.feature_importances_[cols[s]].sum() for s in SENSORS}
    ranked = sorted(SENSORS, key=lambda s: -imp[s])
    print("most relied-on sensors:", ranked[:8], flush=True)

    print("fitting consistency check ...", flush=True)
    cc = ConsistencyCheck().fit(X[idx_fit], X[idx_cal])

    te_fault = idx_te[y[idx_te] == 1]
    te_norm = idx_te[y[idx_te] == 0]
    tr_fault = idx_tr[y[idx_tr] == 1]
    rng = np.random.default_rng(42)

    print("fitting per-sensor voters ...", flush=True)
    rv = ResilientVote().fit(X[idx_fit], y[idx_fit], X[idx_cal], y[idx_cal])
    vote_rows = []

    def log_votes(attacker, k, vf, vn):
        for f in FS:
            vote_rows.append(dict(attacker=attacker, k=k, f=f,
                                  dependability=(vf >= f + 1).mean(), false_trip=(vn >= f + 1).mean()))

    log_votes("none", 0, rv.votes(X[te_fault]), rv.votes(X[te_norm]))

    # clean baseline
    flags_clean, Xr_clean = cc.check_and_repair(X[idx_te])
    rows = [dict(attack="none", attacker="-", k=0,
                 dependability=(det.predict(X[te_fault]) == 1).mean(),
                 false_trip=(det.predict(X[te_norm]) == 1).mean(),
                 dependability_defended=(det.predict(Xr_clean[y[idx_te] == 1]) == 1).mean(),
                 false_trip_defended=(det.predict(Xr_clean[y[idx_te] == 0]) == 1).mean(),
                 attack_flagged=flags_clean.any(1).mean())]
    print(rows[-1], flush=True)

    for attacker in ["random", "targeted"]:
        for k in KS:
            pick = lambda n: [list(rng.choice(SENSORS, k, replace=False)) if attacker == "random" else ranked[:k] for _ in range(n)]
            # masking a real fault
            sens = pick(len(te_fault))
            Xa = feats(attack(ph, te_fault, sens, "mask"))
            fa, Xra = cc.check_and_repair(Xa)
            dep, dep_d = (det.predict(Xa) == 1).mean(), (det.predict(Xra) == 1).mean()
            fl_m = fa.any(1).mean()
            # spoofing a fault during normal operation (donor = a recorded fault, same layout)
            sens = pick(len(te_norm))
            donor = np.array([rng.choice(tr_fault[meta.topo.values[tr_fault] == t]) for t in meta.topo.values[te_norm]])
            Xs = feats(attack(ph, te_norm, sens, "spoof", donor_ph=ph, donor_idx=donor))
            fs, Xrs = cc.check_and_repair(Xs)
            ft, ft_d = (det.predict(Xs) == 1).mean(), (det.predict(Xrs) == 1).mean()
            log_votes(attacker, k, rv.votes(Xa), rv.votes(Xs))
            rows.append(dict(attack="mask+spoof", attacker=attacker, k=k, dependability=dep, false_trip=ft,
                             dependability_defended=dep_d, false_trip_defended=ft_d,
                             attack_flagged=(fl_m + fs.any(1).mean()) / 2))
            print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rows[-1].items()}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fdia.csv"), index=False)
    vdf = pd.DataFrame(vote_rows)
    vdf.to_csv(os.path.join(OUT, "fdia_resilient_vote.csv"), index=False)
    print(vdf.pivot_table(index=["attacker", "k"], columns="f", values=["false_trip", "dependability"]).round(3).to_string())
    with open(os.path.join(OUT, "fdia_sensor_ranking.json"), "w") as f:
        json.dump({s: float(imp[s]) for s in ranked}, f, indent=2)


if __name__ == "__main__":
    run()
