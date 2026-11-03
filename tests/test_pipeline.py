"""Quick checks: python -m pytest tests -q  (about 30 s)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import opendssdirect as dss
from sim import Simulator, TOPOLOGIES, compile_feeder, compute_zones, FAULT_TYPES
from features import build, feature_names, sensor_columns
from topo_aware import signatures


def test_every_layout_converges_and_energises_all_buses():
    for t in TOPOLOGIES:
        compile_feeder(t)
        assert dss.Solution.Converged(), t
        vm = np.array(dss.Circuit.AllBusMagPu())
        assert vm.min() > 0.9, f"{t}: de-energised bus (min {vm.min():.3f} pu)"


def test_five_zones_cover_feeder():
    z = compute_zones()
    assert sorted(set(z.values())) == ["Z1", "Z2", "Z3", "Z4", "Z5"]


def test_samples_and_features_shapes():
    sim = Simulator(0)
    out = [sim.sample("T0_base", ft) for ft in FAULT_TYPES]
    assert all(o is not None for o in out)
    v0 = np.stack([o.v_pre for o in out]); v1 = np.stack([o.v_post for o in out])
    i0 = np.stack([o.i_pre for o in out]); i1 = np.stack([o.i_post for o in out])
    X = build(v0, v1, i0, i1)
    assert X.shape == (5, len(feature_names()))
    assert max(max(c) for c in sensor_columns().values()) == X.shape[1] - 1
    # a bolted three-phase fault must raise the substation current
    lll = out[FAULT_TYPES.index("LLL")]
    assert np.abs(lll.i_post[0]).max() > np.abs(lll.i_pre[0]).max()


def test_reverse_feed_shows_negative_direction():
    # In T2, Sw3 is open and zone Z2 is fed backwards through Sw7 (151 <- 300).
    sig = signatures("T2_sw7_sw3")
    from sim import I_SENSORS
    assert sig["47"][I_SENSORS.index("Sw7")] == -1
    assert sig["47"][I_SENSORS.index("Sw3")] == 0
