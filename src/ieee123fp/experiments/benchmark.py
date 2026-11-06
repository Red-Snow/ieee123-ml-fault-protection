"""Experiment 1: black-box detection, fault-type and zone models under two test protocols.

  random - stratified 80/20 split, all grid layouts mixed (the usual, optimistic test)
  loto   - leave-one-topology-out: train on 4 layouts, test on the 5th, which the model has
           never seen (the honest test for a reconfigurable grid)

Writes: metrics.csv, predictions_loto.csv.gz, summary.json
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ..config import Paths
from ..dataset import load
from ..models.baselines import model_zoo, protection_metrics

log = logging.getLogger(__name__)

TASKS = {
    # task -> (row filter, label column)
    "detection": (lambda m: np.ones(len(m), bool), "is_fault"),
    "type": (lambda m: m.is_fault.values == 1, "ftype"),
    "zone": (lambda m: m.is_fault.values == 1, "zone"),
}


def splits(meta: pd.DataFrame, y: np.ndarray):
    """Yield (protocol, held_out, train_idx, test_idx): one random split, then one per layout."""
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    yield "random", "all", tr, te
    for t in sorted(meta.topo.unique()):
        yield "loto", t, np.where(meta.topo != t)[0], np.where(meta.topo == t)[0]


def run(paths: Paths) -> pd.DataFrame:
    paths.ensure()
    X, meta, _ = load(paths.data)
    log.info("dataset %s %s", X.shape, meta.ftype.value_counts().to_dict())
    rows, preds = [], []
    for task, (filt, col) in TASKS.items():
        mask = filt(meta)
        Xt, mt = X[mask], meta[mask].reset_index(drop=True)
        y = mt[col].values
        task_splits = list(splits(mt, y))
        for mname, make in model_zoo().items():
            for proto, held, tr, te in task_splits:
                t0 = time.time()
                yp = make().fit(Xt[tr], y[tr]).predict(Xt[te])
                r = dict(task=task, model=mname, protocol=proto, held_out=held,
                         acc=accuracy_score(y[te], yp), f1_macro=f1_score(y[te], yp, average="macro"),
                         n_test=len(te), secs=round(time.time() - t0, 1))
                if task == "detection":
                    r["dependability"], r["security"] = protection_metrics(y[te], yp)
                rows.append(r)
                log.info("%-9s %-13s %-6s %-11s acc=%.4f f1=%.4f (%ss)",
                         task, mname, proto, held, r["acc"], r["f1_macro"], r["secs"])
                if proto == "loto":
                    p = mt.iloc[te][["topo", "ftype", "zone", "bus", "rf", "pv_pen", "load_mult"]].copy()
                    p["task"], p["model"], p["y_true"], p["y_pred"] = task, mname, y[te], yp
                    preds.append(p)

    df = pd.DataFrame(rows)
    df.to_csv(paths.results / "metrics.csv", index=False)
    pd.concat(preds).to_csv(paths.results / "predictions_loto.csv.gz", index=False)
    summ: dict = {}
    for (task, model, proto), g in df.groupby(["task", "model", "protocol"]):
        summ.setdefault(task, {}).setdefault(model, {})[proto] = {
            "acc_mean": round(g.acc.mean(), 4), "acc_min": round(g.acc.min(), 4),
            "f1_macro_mean": round(g.f1_macro.mean(), 4)}
    with open(paths.results / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    return df
