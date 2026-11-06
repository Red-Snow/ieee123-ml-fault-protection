"""Fast checks of the grid model, simulator and features (OpenDSS, a few seconds)."""
import numpy as np
import opendssdirect as dss
import pytest

from ieee123fp.config import FAULT_TYPES, I_SENSORS, SENSORS, ZONES
from ieee123fp.features import build, feature_names, sensor_columns
from ieee123fp.grid import TOPOLOGIES, compile_feeder, compute_zones, faultable_buses, path_signatures
from ieee123fp.simulation import Simulator

pytestmark = pytest.mark.needs_feeder


def test_every_layout_converges_and_energises_all_buses():
    for t in TOPOLOGIES:
        compile_feeder(t)
        assert dss.Solution.Converged(), t
        vm = np.array(dss.Circuit.AllBusMagPu())
        assert vm.min() > 0.9, f"{t}: de-energised bus (min {vm.min():.3f} pu)"


def test_five_zones_cover_every_faultable_bus():
    z = compute_zones()
    assert sorted(set(z.values())) == ZONES
    compile_feeder("T0_base")
    assert set(faultable_buses()) <= set(z)


def test_samples_and_features_shapes():
    sim = Simulator(0)
    out = [sim.sample("T0_base", ft) for ft in FAULT_TYPES]
    assert all(o is not None for o in out)
    X = build(*(np.stack([getattr(o, k) for o in out]) for k in ("v_pre", "v_post", "i_pre", "i_post")))
    assert X.shape == (5, len(feature_names())) == (5, 208)
    assert max(max(c) for c in sensor_columns().values()) == X.shape[1] - 1
    assert list(sensor_columns()) == SENSORS
    # a bolted three-phase fault must raise the substation current
    lll = out[FAULT_TYPES.index("LLL")]
    assert np.abs(lll.i_post[0]).max() > np.abs(lll.i_pre[0]).max()


def test_simulator_is_deterministic():
    a, b = Simulator(7).sample("T3_loop", "SLG"), Simulator(7).sample("T3_loop", "SLG")
    assert (a.bus, a.phases, a.rf) == (b.bus, b.phases, b.rf)
    np.testing.assert_array_equal(a.i_post, b.i_post)


def test_reverse_feed_shows_negative_direction():
    # In T2, Sw3 is open and zone Z2 is fed backwards through Sw7 (151 <- 300).
    sig = path_signatures("T2_sw7_sw3")
    assert sig["47"][I_SENSORS.index("Sw7")] == -1
    assert sig["47"][I_SENSORS.index("Sw3")] == 0


def test_substation_sensor_sees_every_downstream_fault_forward():
    # L115 runs 149 -> 1, so every bus except 149 itself lies behind it
    for t in TOPOLOGIES:
        sig = path_signatures(t)
        assert all(s[I_SENSORS.index("L115")] == 1 for b, s in sig.items() if b != "149"), t
