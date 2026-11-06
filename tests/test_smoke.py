"""End-to-end smoke test on a tiny simulated dataset:  pytest -m slow"""
import json

import numpy as np
import pandas as pd
import pytest

from ieee123fp import figures
from ieee123fp.config import ZONES, Paths
from ieee123fp.dataset import generate, load
from ieee123fp.experiments import benchmark, fdia, zone_loto
from ieee123fp.models import ProtectionScheme

pytestmark = [pytest.mark.slow, pytest.mark.needs_feeder]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    p = Paths(root / "data" / "raw_phasors.npz", root / "results", root / "figures").ensure()
    generate(p, per_topo=120, chunks=2, workers=2)
    return p


def test_dataset(paths):
    ds = load(paths.data)
    assert ds.X.shape == (600, 208) and np.isfinite(ds.X).all()
    assert set(ds.meta.topo.value_counts()) == {120}


def test_benchmark(paths):
    df = benchmark.run(paths)
    assert len(df) == 3 * 3 * 6
    assert json.loads((paths.results / "summary.json").read_text()).keys() == {"detection", "type", "zone"}


def test_zone(paths):
    df = zone_loto.run(paths)
    assert len(df) == 5 and df.acc.between(0, 1).all()


def test_fdia(paths):
    fdia.run(paths)
    v = pd.read_csv(paths.results / "fdia_resilient_vote.csv")
    assert {"dependability", "false_trip"} <= set(v.columns)


def test_figures(paths):   # uses the results written by the three tests above
    figures.run(paths)
    assert {f.name for f in paths.figures.iterdir()} >= {"feeder_map.svg", "fdia_false_trips.svg"}


def test_protection_scheme(paths):
    ds = load(paths.data)
    scheme = ProtectionScheme(f=2).fit(ds)
    d = scheme.decide(ds.X[:50], {k: v[:50] for k, v in ds.ph.items()}, topo="T3_loop")
    assert list(d.columns) == ["votes", "trip", "ftype", "zone", "zone_conf"]
    assert set(d.zone[d.trip]) <= set(ZONES) and (d.zone[~d.trip] == "NONE").all()
