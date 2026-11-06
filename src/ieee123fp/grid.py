"""The IEEE 123-node feeder: grid layouts, the OpenDSS model, its graph and protection zones.

Everything that depends on *where things are on the grid* lives here, so the simulator,
the topology-aware model and the figures all share one graph builder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache, lru_cache

import networkx as nx
import numpy as np
import opendssdirect as dss

from .config import FEEDER_DIR, I_SENSORS, SOURCE_MVASC1, SOURCE_MVASC3, SUBSTATION_BUS


# --------------------------------------------------------------------------
# Grid layouts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Topology:
    """One switching state of the feeder.

    close: tie switches to close, as {line: bus it now connects to}
    open:  lines to open
    """

    name: str
    description: str
    close: dict[str, str] = field(default_factory=dict)
    open: tuple[str, ...] = ()

    def commands(self) -> list[str]:
        return ([f"Edit Line.{sw} Bus2={bus}" for sw, bus in self.close.items()]
                + [f"Open Line.{line} 1" for line in self.open])

    @property
    def opened(self) -> set[str]:
        return {line.lower() for line in self.open}


_LAYOUTS = [
    Topology("T0_base", "standard IEEE configuration"),
    Topology("T1_sw7_sw5", "close tie Sw7, open Sw5: zone Z5 is fed from the Z2 side",
             close={"Sw7": "300"}, open=("Sw5",)),
    Topology("T2_sw7_sw3", "close tie Sw7, open Sw3: zone Z2 is fed backwards through Z5",
             close={"Sw7": "300"}, open=("Sw3",)),
    Topology("T3_loop", "close tie Sw7: the feeder becomes a loop",
             close={"Sw7": "300"}),
    Topology("T4_sw8_l93", "close single-phase tie Sw8, open L93: bus 94 is fed from bus 54",
             close={"Sw8": "94.1"}, open=("L93",)),
]
# All five keep every bus energised (checked in tests). Insertion order = study order.
TOPOLOGIES: dict[str, Topology] = {t.name: t for t in _LAYOUTS}

# Switches that bound the five protection zones in the base layout, and the normally-open ties
ZONE_BOUNDARIES = ("sw2", "sw3", "sw4", "sw5")
TIE_SWITCHES = ("sw7", "sw8")
# One bus inside each zone, ordered by distance from the substation: names the zones
ZONE_SEEDS = {"149": "Z1", "135": "Z2", "152": "Z3", "160": "Z4", "197": "Z5"}


class FeederMissing(FileNotFoundError):
    pass


# --------------------------------------------------------------------------
# OpenDSS model
# --------------------------------------------------------------------------
def compile_feeder(topo: str | Topology) -> None:
    """Load the feeder into OpenDSS in the given layout and solve one power flow."""
    t = TOPOLOGIES[topo] if isinstance(topo, str) else topo
    master = FEEDER_DIR / "IEEE123Master.dss"
    if not master.exists():
        raise FeederMissing(f"{master} not found. Download it first:  ieee123fp fetch  (or bash feeder/fetch_feeder.sh)")
    cwd = os.getcwd()
    os.chdir(FEEDER_DIR)
    try:
        dss.Text.Command("Clear")
        dss.Text.Command("Redirect IEEE123Master.dss")
    finally:
        os.chdir(cwd)
    # finite source strength so fault currents are realistic
    dss.Text.Command(f"Edit Vsource.source MVAsc3={SOURCE_MVASC3} MVAsc1={SOURCE_MVASC1}")
    for cmd in t.commands():
        dss.Text.Command(cmd)
    dss.Text.Command("Set VoltageBases=[4.16, 0.48]")
    dss.Text.Command("CalcVoltageBases")
    # loads keep converging when voltage collapses near a fault
    dss.Text.Command("BatchEdit Load..* Vminpu=0.6")
    dss.Solution.Solve()


def faultable_buses() -> dict[str, list[int]]:
    """Medium-voltage (4.16 kV) buses of the compiled circuit and the phases present at each."""
    out = {}
    for b in dss.Circuit.AllBusNames():
        if b in (SUBSTATION_BUS, "150r", "610", "sourcebus") or "open" in b:
            continue
        dss.Circuit.SetActiveBus(b)
        if dss.Bus.kVBase() < 1.0:
            continue
        nodes = [n for n in dss.Bus.Nodes() if n in (1, 2, 3)]
        if nodes:
            out[b] = nodes
    return out


def element_buses(name: str) -> list[str]:
    """Bus names (without node suffixes, lower case) of an OpenDSS element, e.g. 'Line.Sw7'."""
    dss.Circuit.SetActiveElement(name)
    return [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]


def feeder_graph(skip: set[str] | frozenset[str] = frozenset()) -> nx.Graph:
    """Undirected bus graph of the *compiled* circuit from enabled lines and transformers.

    Each edge keeps `els = {element: (bus1, bus2)}`, the element's own reference
    orientation, so a path through it can be read as forward (+1) or backward (-1).
    Elements in `skip` and dangling open-switch buses (*_open) are left out."""
    g = nx.Graph()
    for name in dss.Circuit.AllElementNames():
        cls, el = name.split(".", 1)
        if cls.lower() not in ("line", "transformer") or el.lower() in skip:
            continue
        dss.Circuit.SetActiveElement(name)
        if not dss.CktElement.Enabled():
            continue
        buses = element_buses(name)
        if any("open" in b for b in buses):
            continue
        u, v = buses[0], buses[1]
        if u == v:
            continue
        if not g.has_edge(u, v):
            g.add_edge(u, v, els={})
        g[u][v]["els"][el.lower()] = (u, v)
    return g


# --------------------------------------------------------------------------
# Protection zones and path signatures
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _zones() -> tuple[tuple[str, str], ...]:
    compile_feeder("T0_base")
    g = feeder_graph(set(ZONE_BOUNDARIES) | set(TIE_SWITCHES))
    zones = {}
    for comp in nx.connected_components(g):
        z = next((ZONE_SEEDS[k] for k in ZONE_SEEDS if k in comp), None)
        if z is not None:
            zones.update({b: z for b in comp})
    return tuple(sorted(zones.items()))


def compute_zones() -> dict[str, str]:
    """Zone of every bus: connected component of the BASE layout after removing the four
    sectionalising switches. Zones are geographic and stay fixed when the layout changes,
    like a utility's protection map. Note: compiles the base layout on first call."""
    return dict(_zones())


@cache
def _signatures(topo: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    t = TOPOLOGIES[topo]
    compile_feeder(t)
    g = feeder_graph(t.opened)
    sens = [s.lower() for s in I_SENSORS]
    out = []
    for b in faultable_buses():
        sig = np.zeros(len(sens), int)
        for path in nx.all_simple_paths(g, SUBSTATION_BUS, b.lower()):
            for u, v in zip(path[:-1], path[1:]):
                for el, (b1, _) in g[u][v]["els"].items():
                    if el in sens:
                        sig[sens.index(el)] = 1 if u == b1 else -1
        out.append((b.lower(), tuple(int(x) for x in sig)))
    return tuple(out)


def path_signatures(topo: str) -> dict[str, np.ndarray]:
    """bus -> vector over I_SENSORS with values in {0, +1, -1} for the given layout:

    0  = the sensor is not on any path from the substation to the bus
    +1 = fault current at that bus passes the sensor in its reference direction (bus1 -> bus2)
    -1 = it passes backwards (e.g. a section now fed from the other end)

    This is what the utility's SCADA switch states give us for free."""
    return {b: np.array(s) for b, s in _signatures(topo)}
