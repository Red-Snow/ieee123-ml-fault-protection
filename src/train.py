"""
Train and evaluate fault detection, fault-type classification and fault-zone
localisation under two test protocols:

  random  - stratified 80/20 split, all grid layouts mixed (the usual, optimistic test)
  loto    - leave-one-topology-out: train on 4 layouts, test on the 5th, which the
            model has never seen (the honest test for a reconfigurable grid)

Outputs: results/metrics.csv, results/predictions_loto.csv, results/summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data import ROOT, load  # noqa: E402

OUT = os.path.join(ROOT, "results")


def models():
    return {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=0),
        "HistGradBoost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4,
                                                                     early_stopping=True, max_iter=300, random_state=0)),
    }


TASKS = {
    # task name -> (row filter, label column)
    "detection": (lambda m: np.ones(len(m), bool), "is_fault"),
    "type": (lambda m: m.is_fault.values == 1, "ftype"),
    "zone": (lambda m: m.is_fault.values == 1, "zone"),
}


def protection_metrics(y_true, y_pred):
    """Dependability = share of real faults detected; security = share of normal events not tripped."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    dep = (y_pred[y_true == 1] == 1).mean()
    sec = (y_pred[y_true == 0] == 0).mean()
    return dep, sec


def run():
    os.makedirs(OUT, exist_ok=True)
    X, meta, _ = load()
    print("dataset", X.shape, meta.ftype.value_counts().to_dict(), flush=True)
    rows, preds = [], []
    topos = sorted(meta.topo.unique())

    for task, (filt, col) in TASKS.items():
        mask = filt(meta)
        Xt, mt = X[mask], meta[mask].reset_index(drop=True)
        y = mt[col].values
        splits = []
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
        splits.append(("random", "all", tr, te))
        for t in topos:
            splits.append(("loto", t, np.where(mt.topo != t)[0], np.where(mt.topo == t)[0]))

        for mname, mk in models().items():
            for proto, held, tr, te in splits:
                t0 = time.time()
                clf = mk().fit(Xt[tr], y[tr])
                yp = clf.predict(Xt[te])
                r = dict(task=task, model=mname, protocol=proto, held_out=held,
                         acc=accuracy_score(y[te], yp), f1_macro=f1_score(y[te], yp, average="macro"),
                         n_test=len(te), secs=round(time.time() - t0, 1))
                if task == "detection":
                    r["dependability"], r["security"] = protection_metrics(y[te], yp)
                rows.append(r)
                print(f"{task:9s} {mname:13s} {proto:6s} {held:11s} acc={r['acc']:.4f} f1={r['f1_macro']:.4f} ({r['secs']}s)", flush=True)
                if proto == "loto":
                    p = mt.iloc[te][["topo", "ftype", "zone", "bus", "rf", "pv_pen", "load_mult"]].copy()
                    p["task"], p["model"], p["y_true"], p["y_pred"] = task, mname, y[te], yp
                    preds.append(p)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(OUT, "predictions_loto.csv.gz"), index=False)

    summ = {}
    for (task, model, proto), g in df.groupby(["task", "model", "protocol"]):
        summ.setdefault(task, {}).setdefault(model, {})[proto] = {
            "acc_mean": round(g.acc.mean(), 4), "acc_min": round(g.acc.min(), 4),
            "f1_macro_mean": round(g.f1_macro.mean(), 4)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    run()
