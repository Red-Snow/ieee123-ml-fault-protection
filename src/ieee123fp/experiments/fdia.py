"""Experiment 3: false-data injection against the ML detector, and two defences.

Detector: gradient boosting on all sensors. Attacks: replay masking + spoofing by a random
or a targeted attacker controlling k sensors (see `attacks.py`). Defences: consistency check
(fails) and f-resilient voting (works); see `models/defences.py`.

Writes: fdia.csv, fdia_resilient_vote.csv, fdia_sensor_ranking.json
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from ..attacks import choose_sensors, replay_attack
from ..config import SENSORS, Paths
from ..dataset import load
from ..features import build_from, sensor_columns
from ..models.defences import ConsistencyCheck, ResilientVote

log = logging.getLogger(__name__)

KS = [1, 2, 3, 4, 6, 8, 12]   # compromised sensors
FS = [0, 1, 2, 3, 4, 6]       # vote tolerance f: trip needs > f sensors


def rank_sensors(X, y) -> dict[str, float]:
    """What a model-aware attacker would target: summed random-forest importance per sensor."""
    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0).fit(X, y)
    cols = sensor_columns()
    imp = {s: float(rf.feature_importances_[cols[s]].sum()) for s in SENSORS}
    return dict(sorted(imp.items(), key=lambda kv: -kv[1]))


def run(paths: Paths) -> pd.DataFrame:
    paths.ensure()
    X, meta, ph = load(paths.data)
    y = meta.is_fault.values
    topo = meta.topo.values
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=meta.ftype, random_state=0)
    idx_fit, idx_cal = train_test_split(idx_tr, test_size=0.2, stratify=meta.ftype.values[idx_tr], random_state=1)

    det = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[idx_tr], y[idx_tr])
    imp = rank_sensors(X[idx_tr], y[idx_tr])
    ranked = list(imp)
    log.info("most relied-on sensors: %s", ranked[:8])

    log.info("fitting consistency check ...")
    cc = ConsistencyCheck().fit(X[idx_fit], X[idx_cal])
    log.info("fitting per-sensor voters ...")
    rv = ResilientVote().fit(X[idx_fit], y[idx_fit], X[idx_cal], y[idx_cal])

    te_fault, te_norm = idx_te[y[idx_te] == 1], idx_te[y[idx_te] == 0]
    tr_fault = idx_tr[y[idx_tr] == 1]
    rng = np.random.default_rng(42)
    trips = lambda Z: (det.predict(Z) == 1).mean()  # noqa: E731
    vote_rows = []

    def log_votes(attacker, k, v_fault, v_norm):
        for f in FS:
            vote_rows.append(dict(attacker=attacker, k=k, f=f,
                                  dependability=(v_fault >= f + 1).mean(), false_trip=(v_norm >= f + 1).mean()))

    # clean baseline
    log_votes("none", 0, rv.votes(X[te_fault]), rv.votes(X[te_norm]))
    flags, Xr = cc.check_and_repair(X[idx_te])
    is_f = y[idx_te] == 1
    rows = [dict(attack="none", attacker="-", k=0, dependability=trips(X[te_fault]), false_trip=trips(X[te_norm]),
                 dependability_defended=trips(Xr[is_f]), false_trip_defended=trips(Xr[~is_f]),
                 attack_flagged=flags.any(1).mean())]

    for attacker in ["random", "targeted"]:
        for k in KS:
            # masking a real fault
            Xm = build_from(replay_attack(ph, te_fault, choose_sensors(attacker, k, len(te_fault), rng, ranked), "mask"))
            fm, Xmr = cc.check_and_repair(Xm)
            # spoofing a fault during normal operation (donor = a recorded fault in the same layout)
            sens = choose_sensors(attacker, k, len(te_norm), rng, ranked)
            donor = np.array([rng.choice(tr_fault[topo[tr_fault] == t]) for t in topo[te_norm]])
            Xs = build_from(replay_attack(ph, te_norm, sens, "spoof", donor_ph=ph, donor_idx=donor))
            fs, Xsr = cc.check_and_repair(Xs)
            log_votes(attacker, k, rv.votes(Xm), rv.votes(Xs))
            rows.append(dict(attack="mask+spoof", attacker=attacker, k=k,
                             dependability=trips(Xm), false_trip=trips(Xs),
                             dependability_defended=trips(Xmr), false_trip_defended=trips(Xsr),
                             attack_flagged=(fm.any(1).mean() + fs.any(1).mean()) / 2))
            log.info("%-8s k=%-2d dependability=%.4f false_trip=%.4f", attacker, k, rows[-1]["dependability"],
                     rows[-1]["false_trip"])

    df = pd.DataFrame(rows)
    df.to_csv(paths.results / "fdia.csv", index=False)
    vdf = pd.DataFrame(vote_rows)
    vdf.to_csv(paths.results / "fdia_resilient_vote.csv", index=False)
    log.info("resilient vote:\n%s", vdf.pivot_table(index=["attacker", "k"], columns="f",
                                                     values=["false_trip", "dependability"]).round(3).to_string())
    with open(paths.results / "fdia_sensor_ranking.json", "w") as f:
        json.dump(imp, f, indent=2)
    return df
