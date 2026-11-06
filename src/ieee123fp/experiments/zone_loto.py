"""Experiment 2: topology-aware zone localisation, leave-one-topology-out.

The per-sensor model is trained on four layouts; the held-out layout is used only through
its graph (switch states), never through its measurements.

Writes: metrics_topology_aware.csv, topology_aware_sensor_acc.json
"""
from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..config import Paths
from ..dataset import load
from ..grid import TOPOLOGIES
from ..models.topology_aware import TopologyAwareZoneLocator

log = logging.getLogger(__name__)


def run(paths: Paths) -> pd.DataFrame:
    paths.ensure()
    ds = load(paths.data).faults()
    results, per_sensor = [], []
    for held in TOPOLOGIES:
        tr, te = ds.subset((ds.meta.topo != held).values), ds.subset((ds.meta.topo == held).values)
        loc = TopologyAwareZoneLocator().fit(tr.X, tr.ph, tr.meta.topo.values, tr.meta.bus.values)
        y_dir = loc.direction_labels(te.meta.topo.values, te.meta.bus.values)
        per_sensor.append(dict(held_out=held,
                               sensor_acc=float((loc.predict_directions(te.X, te.ph) == y_dir).mean())))
        pred, yt = loc.predict(te.X, te.ph, topo=held), te.meta.zone.values
        results.append(dict(task="zone", model="TopologyAware-HGB", protocol="loto", held_out=held,
                            acc=accuracy_score(yt, pred), f1_macro=f1_score(yt, pred, average="macro"),
                            n_test=len(te)))
        log.info("held out %-11s zone acc=%.4f  per-sensor direction acc=%.4f",
                 held, results[-1]["acc"], per_sensor[-1]["sensor_acc"])

    df = pd.DataFrame(results)
    df.to_csv(paths.results / "metrics_topology_aware.csv", index=False)
    with open(paths.results / "topology_aware_sensor_acc.json", "w") as fh:
        json.dump(per_sensor, fh, indent=2)
    return df
