"""
Fault simulation on the IEEE 123-node test feeder (OpenDSS, phasor snapshot).

For every sample we:
  1. pick a grid layout (topology), a load level, a PV (solar) penetration and a
     capacitor state;
  2. solve a "before" power flow with voltage regulators active;
  3. freeze regulator taps (they are far too slow to react during a fault),
     apply an event -- either a fault or, for normal samples, a load/capacitor
     switching disturbance -- and solve the "after" state;
  4. read three-phase voltage phasors at 15 buses and three-phase current
     phasors at 8 line sensors (think micro-PMUs / numerical relays), before and
     after the event.

Labels: fault yes/no, fault type (SLG, LL, LLG, LLL), fault zone (Z1..Z5).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import opendssdirect as dss

FEEDER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feeder")

# --------------------------------------------------------------------------
# Grid layouts. All five keep every bus energised (checked in tests).
# --------------------------------------------------------------------------
TOPOLOGIES: dict[str, list[str]] = {
    # T0: the standard radial configuration from the IEEE test case
    "T0_base": [],
    # T1: tie switch Sw7 closed, Sw5 opened -> buses 197-114 fed from the 151/300 side
    "T1_sw7_sw5": ["Edit Line.Sw7 Bus2=300", "Open Line.Sw5 1"],
    # T2: tie switch Sw7 closed, Sw3 opened -> buses 135-151 fed backwards via 300
    "T2_sw7_sw3": ["Edit Line.Sw7 Bus2=300", "Open Line.Sw3 1"],
    # T3: tie switch Sw7 closed with everything else closed -> a meshed (looped) feeder
    "T3_loop": ["Edit Line.Sw7 Bus2=300"],
    # T4: single-phase tie Sw8 closed, line L93 opened -> bus 94 fed from bus 54
    "T4_sw8_l93": ["Edit Line.Sw8 Bus2=94.1", "Open Line.L93 1"],
}

# Switches that bound the five protection zones in the base layout
ZONE_BOUNDARIES = ["sw2", "sw3", "sw4", "sw5"]

# Measurement points (fixed hardware: same sensors in every layout)
V_SENSORS = ["149", "13", "18", "35", "52", "60", "67", "76", "97", "101", "108", "300", "151", "450", "83"]
I_SENSORS = ["L115", "Sw2", "Sw3", "Sw4", "Sw5", "Sw7", "L108", "L67"]

FAULT_TYPES = ["NORMAL", "SLG", "LL", "LLG", "LLL"]
# sampling weights for each class
CLASS_WEIGHTS = {"NORMAL": 0.20, "SLG": 0.32, "LL": 0.16, "LLG": 0.16, "LLL": 0.16}

THREE_PHASE_BUSES_FOR_PV = None  # filled lazily

NOMINAL_ANG = np.array([0.0, -120.0, 120.0])


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


# --------------------------------------------------------------------------
def compile_feeder(topo: str) -> None:
    cwd = os.getcwd()
    os.chdir(FEEDER_DIR)
    try:
        dss.Text.Command("Clear")
        dss.Text.Command("Redirect IEEE123Master.dss")
    finally:
        os.chdir(cwd)
    # finite source strength so fault currents are realistic (~28 kA 3ph at 4.16 kV)
    dss.Text.Command("Edit Vsource.source MVAsc3=200 MVAsc1=210")
    for cmd in TOPOLOGIES[topo]:
        dss.Text.Command(cmd)
    dss.Text.Command("Set VoltageBases=[4.16, 0.48]")
    dss.Text.Command("CalcVoltageBases")
    # loads keep converging when voltage collapses near a fault
    dss.Text.Command("BatchEdit Load..* Vminpu=0.6")
    dss.Solution.Solve()


def _bus_graph(skip_lines: set[str]) -> dict[str, set[str]]:
    """Undirected bus graph from all two-terminal power-delivery elements."""
    g: dict[str, set[str]] = {}
    for name in dss.Circuit.AllElementNames():
        cls, el = name.split(".", 1)
        if cls.lower() not in ("line", "transformer"):
            continue
        if el.lower() in skip_lines:
            continue
        dss.Circuit.SetActiveElement(name)
        buses = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
        if not dss.CktElement.Enabled():
            continue
        for a in buses:
            g.setdefault(a, set())
        for a in buses:
            for b in buses:
                if a != b:
                    g[a].add(b)
    return g


def compute_zones() -> dict[str, str]:
    """Zone of every bus = connected component of the BASE layout after
    removing the four sectionalising switches. Zones are geographic and stay
    fixed when the layout changes, like a utility's protection map."""
    compile_feeder("T0_base")
    g = _bus_graph(set(ZONE_BOUNDARIES) | {"sw7", "sw8"})
    seen: dict[str, str] = {}
    comps = []
    for start in g:
        if start in seen:
            continue
        stack, comp = [start], []
        seen[start] = "?"
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in g[u]:
                if v not in seen:
                    seen[v] = "?"
                    stack.append(v)
        comps.append(comp)
    # name zones by distance from the substation: the one with 149 is Z1
    order = {"149": 1, "135": 2, "152": 3, "160": 4, "197": 5}
    zones: dict[str, str] = {}
    for comp in comps:
        z = next((f"Z{order[k]}" for k in order if k in comp), None)
        if z is None:  # dummy open-switch buses
            continue
        for b in comp:
            zones[b] = z
    return zones


def faultable_buses() -> dict[str, list[int]]:
    """Medium-voltage buses (4.16 kV) and the phases present at each."""
    out = {}
    for b in dss.Circuit.AllBusNames():
        if b in ("150", "150r", "610", "sourcebus") or "open" in b:
            continue
        dss.Circuit.SetActiveBus(b)
        if dss.Bus.kVBase() < 1.0:
            continue
        nodes = [n for n in dss.Bus.Nodes() if n in (1, 2, 3)]
        if nodes:
            out[b] = nodes
    return out


def _read_v() -> np.ndarray:
    out = np.zeros((len(V_SENSORS), 3), dtype=complex)
    for k, b in enumerate(V_SENSORS):
        dss.Circuit.SetActiveBus(b)
        nodes = dss.Bus.Nodes()
        pu = np.array(dss.Bus.PuVoltage()).reshape(-1, 2)
        for n, (re, im) in zip(nodes, pu):
            if n in (1, 2, 3):
                out[k, n - 1] = complex(re, im)
    return out


def _read_i() -> np.ndarray:
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


class Simulator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.zones = compute_zones()
        self.topo = None

    # ------------------------------------------------------------------
    def _setup(self, topo: str):
        if topo != self.topo or getattr(self, 'n_extra', 0) > 300:
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

    def _new(self, cmd: str):
        self.n_extra += 1
        dss.Text.Command(cmd.format(n=self.n_extra))
        return self.n_extra

    # ------------------------------------------------------------------
    def sample(self, topo: str, ftype: str) -> Sample | None:
        r = self.rng
        self._setup(topo)
        dss.Text.Command("Set ControlMode=static")

        load_mult = r.uniform(0.4, 1.2)
        self._set_loads(load_mult)
        cap_on = int(r.random() < 0.7)
        dss.Text.Command(f"Edit Capacitor.C83 enabled={'yes' if cap_on else 'no'}")

        # Solar (PV): simplified as generators that inject constant power and switch to a
        # constant impedance below 0.5 pu voltage (real inverters limit fault current to
        # ~1.1-1.5 pu; see README limitations)
        pv_names = []
        pv_pen = 0.0
        if r.random() < 0.7:
            pv_pen = r.uniform(0.05, 0.6)
            k = r.randint(1, 5)
            sites = r.sample(self.bus3, k)
            for s in sites:
                kw = pv_pen * self.total_kw / k
                n = self._new(
                    f"New Generator.pv{{n}} bus1={s} phases=3 kV=4.16 kW={kw:.1f} pf=1 model=1 Vminpu=0.5 Vmaxpu=1.2"
                )
                pv_names.append(f"pv{n}")

        dss.Solution.Solve()
        if not dss.Solution.Converged():
            self._cleanup(pv_names, [])
            return None
        v_pre, i_pre = _read_v(), _read_i()

        # freeze regulator taps for the event
        dss.Text.Command("Set ControlMode=off")
        extra = []
        bus, phases, rf = "-", "-", 0.0
        if ftype == "NORMAL":
            # a harmless disturbance: load swing and/or capacitor switching
            self._set_loads(load_mult * r.uniform(0.8, 1.3), jitter=0.05)
            if r.random() < 0.5:
                dss.Text.Command(f"Edit Capacitor.C83 enabled={'no' if cap_on else 'yes'}")
            if r.random() < 0.4:  # a large single motor/load start somewhere
                b = r.choice(self.bus3)
                n = self._new(f"New Load.step{{n}} bus1={b} phases=3 kV=4.16 kW={r.uniform(100, 400):.0f} kvar={r.uniform(50, 300):.0f} model=2 Vminpu=0.6")
                extra.append(f"Load.step{n}")
            zone = "NONE"
        else:
            need = {"SLG": 1, "LL": 2, "LLG": 2, "LLL": 3}[ftype]
            cands = [b for b, nd in self.buses.items() if len(nd) >= need]
            bus = r.choice(cands)
            ph = sorted(r.sample(self.buses[bus], need))
            phases = "".join("ABC"[p - 1] for p in ph)
            if ftype == "SLG":
                rf = 10 ** r.uniform(np.log10(0.05), np.log10(40))
                cmd = f"New Fault.f{{n}} bus1={bus}.{ph[0]} phases=1 r={rf:.4f}"
            elif ftype == "LL":
                rf = 10 ** r.uniform(-2, np.log10(3))
                cmd = f"New Fault.f{{n}} bus1={bus}.{ph[0]} bus2={bus}.{ph[1]} phases=1 r={rf:.4f}"
            elif ftype == "LLG":
                rf = 10 ** r.uniform(np.log10(0.05), np.log10(20))
                cmd = f"New Fault.f{{n}} bus1={bus}.{ph[0]}.{ph[1]} phases=2 r={rf:.4f}"
            else:
                rf = 10 ** r.uniform(-2, np.log10(3))
                cmd = f"New Fault.f{{n}} bus1={bus}.1.2.3 phases=3 r={rf:.4f}"
            n = self._new(cmd)
            extra.append(f"Fault.f{n}")
            zone = self.zones[bus.lower()]

        dss.Solution.Solve()
        ok = dss.Solution.Converged()
        v_post, i_post = _read_v(), _read_i()
        self._cleanup(pv_names, extra)
        if not ok:
            return None
        return Sample(topo, ftype, zone, bus, phases, rf, load_mult, pv_pen, cap_on,
                      v_pre, v_post, i_pre, i_post)

    def _cleanup(self, pv_names, extra):
        for p in pv_names:
            dss.Text.Command(f"Edit Generator.{p} enabled=no")
        for e in extra:
            dss.Text.Command(f"Edit {e} enabled=no")
        dss.Text.Command("Edit Capacitor.C83 enabled=yes")


def sample_class(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for k, w in CLASS_WEIGHTS.items():
        acc += w
        if x < acc:
            return k
    return "LLL"
