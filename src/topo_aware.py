"""
Topology-aware (physics-informed) fault-zone localisation.

Why: a black-box classifier maps "this pattern of measurements" -> "zone".
When the grid is reconfigured, the same zone produces a different pattern
(fault current now arrives through a different switch), so the mapping breaks.

Idea, in two steps:
  1. A per-sensor ML model answers a LOCAL question that means the same thing in
     every layout: "is fault current flowing through me, and which way?"
     (forward / backward / not at all). Its only inputs are the sensor's own
     incremental current projected onto the substation's incremental current,
     and the size of its change relative to the largest change on the feeder.
  2. The current switch states -- which the utility's SCADA already knows --
     give the grid graph. For every bus we know which sensors sit on the path(s)
     from the substation to that bus, and in which direction. We pick the zone
     whose buses best explain the per-sensor answers (a likelihood score over
     buses, summed per zone).

The model never sees the held-out layout during training; only the graph does.
"""
from __future__ import annotations

import json
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
import opendssdirect as dss  # noqa: E402

from data import ROOT, ZONES, load  # noqa: E402
from features import sensor_columns  # noqa: E402
from sim import I_SENSORS, TOPOLOGIES, compile_feeder, compute_zones, faultable_buses  # noqa: E402

OUT = os.path.join(ROOT, "results")


def topology_graph(topo: str) -> nx.Graph:
    compile_feeder(topo)
    opened = {c.split()[1].split(".")[1].lower() for c in TOPOLOGIES[topo] if c.lower().startswith("open")}
    g = nx.Graph()
    for name in dss.Circuit.AllElementNames():
        cls, el = name.split(".", 1)
        if cls.lower() not in ("line", "transformer") or el.lower() in opened:
            continue
        dss.Circuit.SetActiveElement(name)
        if not dss.CktElement.Enabled():
            continue
        buses = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
        if any("open" in b for b in buses):
            continue
        u, v = buses[0], buses[1]
        if u == v:
            continue
        if not g.has_edge(u, v):
            g.add_edge(u, v, els={})
        g[u][v]["els"][el.lower()] = (u, v)  # element's own bus1 -> bus2 orientation
    return g


def signatures(topo: str) -> dict[str, np.ndarray]:
    """bus -> vector over I_SENSORS with values in {0, +1, -1}:
    0  = sensor is not on any path from the substation to the faulted bus
    +1 = fault current passes the sensor in its reference direction (its bus1 -> bus2)
    -1 = fault current passes it backwards (e.g. a feeder section now fed from the other end)."""
    g = topology_graph(topo)
    sens = [s.lower() for s in I_SENSORS]
    buses = faultable_buses()
    out = {}
    for b in buses:
        sig = np.zeros(len(sens), int)
        for path in nx.all_simple_paths(g, "150", b.lower()):
            for u, v in zip(path[:-1], path[1:]):
                for el, (b1, b2) in g[u][v]["els"].items():
                    if el in sens:
                        sig[sens.index(el)] = 1 if (u, v) == (b1, b2) else -1
        if b.lower() == "150":
            continue
        out[b.lower()] = sig
    return out


def local_rows(X: np.ndarray, ph: dict, flip: bool = False) -> np.ndarray:
    """(n, F) -> (n * n_sensors, 4) local features per current sensor:
    Re/Im/|.| of its incremental current projected on the substation's, and its
    largest phase change relative to the largest change on the feeder.

    No sensor identity is included, so every sensor shares one "which way is the
    fault current flowing?" rule. flip=True gives the same event as seen by a
    sensor wired backwards: every current phasor rotates by 180 deg, so the
    projection changes sign and the direction label flips. This is an exact
    physical symmetry, used as data augmentation."""
    cols = sensor_columns()
    blocks = [X[:, cols[f"I{s}"]] for s in I_SENSORS]          # each (n, 11)
    dmax = [np.abs(b[:, 6:9]).max(1) for b in blocks]
    big = np.maximum(np.max(np.stack(dmax, 1), 1), 1e-6)
    dI = (ph["i_post"] - ph["i_pre"]) / 1000.0                  # incremental (superimposed) current, kA
    # Direction is judged against the substation breaker's own incremental current
    # (sensor L115, always fed from the source) -- the same polarising idea a
    # directional relay uses. This removes the dependence on fault resistance/type.
    # Projection of each sensor's incremental current on the substation's, weighted by
    # the substation current itself (so phases the fault does not touch contribute ~0):
    #   proj_k = sum_p dI_k,p * conj(ref_p) / sum_p |ref_p|^2
    # Re(proj) > 0: fault current flows the same way as at the substation (forward),
    # Re(proj) < 0: it flows backwards through this sensor, ~0: sensor not on the fault path.
    ref = dI[:, 0, :]
    proj = (dI * np.conj(ref)[:, None, :]).sum(2) / np.maximum((np.abs(ref) ** 2).sum(1), 1e-9)[:, None]
    if flip:
        proj = -proj
    rows = []
    for k in range(len(blocks)):
        # Only physically meaningful, layout-independent quantities are kept. Raw
        # magnitudes/angles were tried first and let the model memorise layout-specific
        # patterns (loop accuracy collapsed) -- see README "What did not work".
        rows.append(np.stack([proj[:, k].real, proj[:, k].imag, np.abs(proj[:, k]), dmax[k] / big], 1))
    return np.stack(rows, 1).reshape(len(X) * len(I_SENSORS), -1)


def run():
    X, meta, ph = load()
    f = meta.is_fault.values == 1
    X, meta = X[f], meta[f].reset_index(drop=True)
    ph = {k: v[f] for k, v in ph.items()}
    zones = compute_zones()
    sigs = {t: signatures(t) for t in TOPOLOGIES}
    S = len(I_SENSORS)
    Y = np.stack([sigs[t][b.lower()] for t, b in zip(meta.topo, meta.bus)])  # (n, S)
    R = local_rows(X, ph)
    R_flip = local_rows(X, ph, flip=True)

    results, per_sensor = [], []
    for held in TOPOLOGIES:
        tr = np.repeat((meta.topo != held).values, S)
        te_s = (meta.topo == held).values
        te = np.repeat(te_s, S)
        y = Y.reshape(-1)
        # Augmentation 1 (direction symmetry): the same event seen by a reversed sensor.
        # Augmentation 2 (current division): in a meshed grid the fault current splits
        # over parallel paths, so an on-path sensor may carry only part of it. Scale the
        # on-path rows' size features by a random factor in [0.05, 1]; labels unchanged.
        rng = np.random.default_rng(0)
        Rd = R[tr].copy()
        k = rng.uniform(0.05, 1.0, len(Rd))
        on = y[tr] != 0
        Rd[on] *= k[on, None]
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(
            np.vstack([R[tr], R_flip[tr], Rd, -Rd * np.array([1, 1, -1, -1])]),
            np.concatenate([y[tr], -y[tr], y[tr], -y[tr]]))
        cls = list(clf.classes_)                                    # [-1, 0, 1]
        P = clf.predict_proba(R[te]).reshape(-1, S, len(cls)).clip(1e-4, 1)
        logP = np.log(P)
        per_sensor.append(dict(held_out=held, sensor_acc=float((np.array(cls)[P.argmax(2)] == Y[te_s]).mean())))

        # decode with the held-out layout's graph
        bus_list = list(sigs[held])
        B = np.stack([sigs[held][b] for b in bus_list])            # (n_bus, S)
        zb = np.array([zones[b] for b in bus_list])
        ll = sum(logP[:, :, i] @ (B == c).T.astype(float) for i, c in enumerate(cls))  # (n, n_bus)
        ll -= ll.max(1, keepdims=True)
        post = np.stack([np.exp(ll[:, zb == z]).sum(1) for z in ZONES], 1)
        pred = np.array(ZONES)[post.argmax(1)]
        yt = meta.zone[te_s].values
        results.append(dict(task="zone", model="TopologyAware-HGB", protocol="loto", held_out=held,
                            acc=accuracy_score(yt, pred), f1_macro=f1_score(yt, pred, average="macro"),
                            n_test=int(te_s.sum())))
        print(results[-1], per_sensor[-1], flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "metrics_topology_aware.csv"), index=False)
    with open(os.path.join(OUT, "topology_aware_sensor_acc.json"), "w") as fh:
        json.dump(per_sensor, fh, indent=2)
    print(df[["held_out", "acc", "f1_macro"]])


if __name__ == "__main__":
    run()
