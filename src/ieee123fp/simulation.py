"""Fault and normal-event simulation on the IEEE 123-node feeder (OpenDSS phasor snapshots).

For every sample we:
  1. pick a load level, a PV (solar) penetration and a capacitor state;
  2. solve a "before" power flow with voltage regulators active;
  3. freeze regulator taps (they are far too slow to react during a fault), apply an
     event -- a fault or, for normal samples, a load/capacitor switching disturbance --
     and solve the "after" state;
  4. read three-phase voltage phasors at 15 buses and three-phase current phasors at
     8 line sensors (think micro-PMUs / numerical relays), before and after the event.

Labels: fault yes/no, fault type (SLG, LL, LLG, LLL), fault zone (Z1..Z5).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np
import opendssdirect as dss

from .config import CAPACITOR, CLASS_WEIGHTS, I_SENSORS, PHASES_NEEDED, RF_RANGE, V_SENSORS
from .grid import compile_feeder, compute_zones, faultable_buses

META_COLUMNS = ["topo", "ftype", "zone", "bus", "phases", "rf", "load_mult", "pv_pen", "cap_on"]


@dataclass
class Sample:
    topo: str
    ftype: str
    zone: str
    bus: str
    phases: str
    rf: float
    load_mult: float
    pv_pen: float
    cap_on: int
    v_pre: np.ndarray = field(repr=False)   # (n_v, 3) complex, pu
    v_post: np.ndarray = field(repr=False)
    i_pre: np.ndarray = field(repr=False)   # (n_i, 3) complex, A
    i_post: np.ndarray = field(repr=False)

    def meta(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in META_COLUMNS}


def read_voltages() -> np.ndarray:
    out = np.zeros((len(V_SENSORS), 3), dtype=complex)
    for k, b in enumerate(V_SENSORS):
        dss.Circuit.SetActiveBus(b)
        pu = np.array(dss.Bus.PuVoltage()).reshape(-1, 2)
        for n, (re, im) in zip(dss.Bus.Nodes(), pu):
            if n in (1, 2, 3):
                out[k, n - 1] = complex(re, im)
    return out


def read_currents() -> np.ndarray:
    out = np.zeros((len(I_SENSORS), 3), dtype=complex)
    for k, el in enumerate(I_SENSORS):
        dss.Circuit.SetActiveElement(f"Line.{el}")
        nodes = dss.CktElement.NodeOrder()
        nc = dss.CktElement.NumConductors()
        cur = np.array(dss.CktElement.Currents()).reshape(-1, 2)[:nc]  # terminal 1
        for n, (re, im) in zip(nodes[:nc], cur):
            if n in (1, 2, 3):
                out[k, n - 1] = complex(re, im)
    return out


def _log_uniform(r: random.Random, lo: float, hi: float) -> float:
    return 10 ** r.uniform(np.log10(lo), np.log10(hi))


class Simulator:
    """Draws random operating points and events for one OpenDSS instance.

    Deterministic for a given seed and call sequence. Temporary OpenDSS objects
    (PV generators, faults, motor loads) are disabled after each sample, and the circuit
    is recompiled every few hundred samples so they don't pile up."""

    RECOMPILE_EVERY = 300

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.zones = compute_zones()
        self.topo: str | None = None
        self.n_extra = 0

    # ------------------------------------------------------------------
    def _setup(self, topo: str):
        if topo == self.topo and self.n_extra <= self.RECOMPILE_EVERY:
            return
        compile_feeder(topo)
        self.topo = topo
        self.buses = faultable_buses()
        self.bus3 = [b for b, n in self.buses.items() if len(n) == 3]
        self.load_names = dss.Loads.AllNames()
        self.base_kw = {}
        for ln in self.load_names:
            dss.Loads.Name(ln)
            self.base_kw[ln] = (dss.Loads.kW(), dss.Loads.kvar())
        self.total_kw = sum(v[0] for v in self.base_kw.values())
        self.n_extra = 0

    def _set_loads(self, mult: float, jitter: float = 0.10):
        for ln in self.load_names:
            kw, kvar = self.base_kw[ln]
            f = mult * (1 + self.rng.uniform(-jitter, jitter))
            dss.Loads.Name(ln)
            dss.Loads.kW(kw * f)
            dss.Loads.kvar(kvar * f)

    def _new(self, cmd: str) -> int:
        self.n_extra += 1
        dss.Text.Command(cmd.format(n=self.n_extra))
        return self.n_extra

    def _set_capacitor(self, on: bool):
        dss.Text.Command(f"Edit Capacitor.{CAPACITOR} enabled={'yes' if on else 'no'}")

    # ------------------------------------------------------------------
    def _add_pv(self) -> tuple[list[str], float]:
        """Solar, simplified as generators that inject constant power and switch to a
        constant impedance below 0.5 pu (real inverters limit fault current to ~1.1-1.5 pu;
        see README limitations)."""
        r = self.rng
        if r.random() >= 0.7:
            return [], 0.0
        pv_pen = r.uniform(0.05, 0.6)
        k = r.randint(1, 5)
        names = []
        for s in r.sample(self.bus3, k):
            kw = pv_pen * self.total_kw / k
            n = self._new(f"New Generator.pv{{n}} bus1={s} phases=3 kV=4.16 kW={kw:.1f} pf=1 "
                          "model=1 Vminpu=0.5 Vmaxpu=1.2")
            names.append(f"pv{n}")
        return names, pv_pen

    def _normal_event(self, load_mult: float, cap_on: int) -> list[str]:
        """A harmless but fault-like disturbance: load swing, capacitor switching, motor start."""
        r = self.rng
        extra = []
        self._set_loads(load_mult * r.uniform(0.8, 1.3), jitter=0.05)
        if r.random() < 0.5:
            self._set_capacitor(not cap_on)
        if r.random() < 0.4:
            b = r.choice(self.bus3)
            n = self._new(f"New Load.step{{n}} bus1={b} phases=3 kV=4.16 kW={r.uniform(100, 400):.0f} "
                          f"kvar={r.uniform(50, 300):.0f} model=2 Vminpu=0.6")
            extra.append(f"Load.step{n}")
        return extra

    def _fault(self, ftype: str) -> tuple[list[str], str, str, float]:
        r = self.rng
        need = PHASES_NEEDED[ftype]
        bus = r.choice([b for b, nd in self.buses.items() if len(nd) >= need])
        ph = sorted(r.sample(self.buses[bus], need))
        phases = "".join("ABC"[p - 1] for p in ph)
        rf = _log_uniform(r, *RF_RANGE[ftype])
        if ftype == "SLG":
            conn = f"bus1={bus}.{ph[0]} phases=1"
        elif ftype == "LL":
            conn = f"bus1={bus}.{ph[0]} bus2={bus}.{ph[1]} phases=1"
        elif ftype == "LLG":
            conn = f"bus1={bus}.{ph[0]}.{ph[1]} phases=2"
        else:
            conn = f"bus1={bus}.1.2.3 phases=3"
        cmd = f"New Fault.f{{n}} {conn} r={rf:.4f}"
        n = self._new(cmd)
        return [f"Fault.f{n}"], bus, phases, rf

    # ------------------------------------------------------------------
    def sample(self, topo: str, ftype: str) -> Sample | None:
        """One event in layout `topo`. Returns None if a power flow did not converge."""
        r = self.rng
        self._setup(topo)
        dss.Text.Command("Set ControlMode=static")

        load_mult = r.uniform(0.4, 1.2)
        self._set_loads(load_mult)
        cap_on = int(r.random() < 0.7)
        self._set_capacitor(cap_on)
        pv_names, pv_pen = self._add_pv()

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            self._cleanup(pv_names, [])
            return None
        v_pre, i_pre = read_voltages(), read_currents()

        dss.Text.Command("Set ControlMode=off")   # freeze regulator taps for the event
        if ftype == "NORMAL":
            extra, bus, phases, rf, zone = self._normal_event(load_mult, cap_on), "-", "-", 0.0, "NONE"
        else:
            extra, bus, phases, rf = self._fault(ftype)
            zone = self.zones[bus.lower()]

        dss.Solution.Solve()
        ok = dss.Solution.Converged()
        v_post, i_post = read_voltages(), read_currents()
        self._cleanup(pv_names, extra)
        if not ok:
            return None
        return Sample(topo, ftype, zone, bus, phases, rf, load_mult, pv_pen, cap_on, v_pre, v_post, i_pre, i_post)

    def _cleanup(self, pv_names, extra):
        for p in pv_names:
            dss.Text.Command(f"Edit Generator.{p} enabled=no")
        for e in extra:
            dss.Text.Command(f"Edit {e} enabled=no")
        self._set_capacitor(True)


def sample_class(rng: random.Random) -> str:
    """Draw an event class with the study's class weights."""
    x, acc = rng.random(), 0.0
    for k, w in CLASS_WEIGHTS.items():
        acc += w
        if x < acc:
            return k
    return k  # guard against float round-off in the cumulative sum
