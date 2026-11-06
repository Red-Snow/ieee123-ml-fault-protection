"""Pure-Python unit tests (no OpenDSS solve needed)."""
import numpy as np

from ieee123fp.attacks import choose_sensors, replay_attack
from ieee123fp.config import I_SENSORS, SENSORS, V_SENSORS
from ieee123fp.dataset import make_jobs
from ieee123fp.grid import TOPOLOGIES


def test_topology_commands():
    t = TOPOLOGIES["T1_sw7_sw5"]
    assert t.commands() == ["Edit Line.Sw7 Bus2=300", "Open Line.Sw5 1"]
    assert t.opened == {"sw5"}
    assert TOPOLOGIES["T0_base"].commands() == []


def test_jobs_keep_every_sample_and_fixed_seeds():
    jobs = make_jobs(6000, 6)
    assert len(jobs) == 30 and {n for _, n, _ in jobs} == {1000}
    assert jobs[7] == ("T1_sw7_sw5", 1000, 1001)
    odd = make_jobs(1003, 6)
    assert sum(n for t, n, _ in odd if t == "T0_base") == 1003


def _phasors(n=4, seed=0):
    rng = np.random.default_rng(seed)
    c = lambda s: rng.normal(size=s) + 1j * rng.normal(size=s)
    nv, ni = (n, len(V_SENSORS), 3), (n, len(I_SENSORS), 3)
    return {"v_pre": c(nv), "v_post": c(nv), "i_pre": c(ni), "i_post": c(ni)}


def test_mask_replays_pre_event_reading_on_chosen_sensors_only():
    ph = _phasors()
    idx = np.array([1, 3])
    a = replay_attack(ph, idx, [["V13", "ISw2"], []], "mask")
    j_v, j_i = V_SENSORS.index("13"), I_SENSORS.index("Sw2")
    np.testing.assert_array_equal(a["v_post"][0, j_v], ph["v_pre"][1, j_v])
    np.testing.assert_array_equal(a["i_post"][0, j_i], ph["i_pre"][1, j_i])
    np.testing.assert_array_equal(a["v_post"][1], ph["v_post"][3])        # untouched row
    np.testing.assert_array_equal(ph["v_post"][1, j_v], _phasors()["v_post"][1, j_v])  # input not mutated


def test_spoof_replays_donor():
    ph, donor = _phasors(), _phasors(seed=1)
    a = replay_attack(ph, np.array([0]), [["IL115"]], "spoof", donor_ph=donor, donor_idx=np.array([2]))
    np.testing.assert_array_equal(a["i_post"][0, 0], donor["i_post"][2, 0])


def test_attacker_sensor_choice():
    rng = np.random.default_rng(0)
    r = choose_sensors("random", 3, 5, rng, SENSORS)
    assert len(r) == 5 and all(len(set(s)) == 3 for s in r)
    assert choose_sensors("targeted", 2, 2, rng, ["a", "b", "c"]) == [["a", "b"], ["a", "b"]]
