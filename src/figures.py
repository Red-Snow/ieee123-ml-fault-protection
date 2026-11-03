"""Make the README figures (SVG) from the saved result files. No retraining needed."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "svg.fonttype": "none", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
TOPO_LABEL = {"T0_base": "T0\nbase", "T1_sw7_sw5": "T1\nSw7 in, Sw5 out", "T2_sw7_sw3": "T2\nSw7 in, Sw3 out",
              "T3_loop": "T3\nlooped", "T4_sw8_l93": "T4\nSw8 in, L93 out"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def zone_gap():
    m = pd.read_csv(os.path.join(RES, "metrics.csv"))
    ta = pd.read_csv(os.path.join(RES, "metrics_topology_aware.csv"))
    z = m[(m.task == "zone") & (m.protocol == "loto")]
    rand = m[(m.task == "zone") & (m.protocol == "random")].acc.max()
    topos = list(TOPO_LABEL)
    series = [("Random forest", z[z.model == "RandomForest"]), ("Gradient boosting", z[z.model == "HistGradBoost"]),
              ("MLP", z[z.model == "MLP"]), ("Topology-aware (this repo)", ta)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    x = np.arange(len(topos))
    for i, (lab, d) in enumerate(series):
        vals = [100 * d[d.held_out == t].acc.values[0] for t in topos]
        ax.bar(x + (i - 1.5) * w, vals, w - 0.03, color=C[i], label=lab, edgecolor="#fcfcfb", linewidth=0)
        if i == 3:
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 1.5) * w, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(100 * rand, color=INK2, lw=1, ls="--")
    ax.text(-0.45, 100 * rand + 1.5, f"Random 80/20 split, all layouts mixed: {100*rand:.1f}%", ha="left", fontsize=8, color=INK2)
    ax.set_xticks(x, [TOPO_LABEL[t] for t in topos], fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Fault-zone accuracy on the unseen layout (%)")
    ax.set_title("Zone localisation when the grid layout was never seen in training", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=4, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "zone_generalisation.svg")


def fdia():
    a = pd.read_csv(os.path.join(RES, "fdia.csv"))
    v = pd.read_csv(os.path.join(RES, "fdia_resilient_vote.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    t = a[a.attacker == "targeted"].sort_values("k")
    base = a[a.attacker == "-"]
    ks = [0] + list(t.k)
    ax.plot(ks, 100 * np.r_[base.false_trip.values, t.false_trip.values], color=C[1], lw=2, marker="o", ms=5,
            label="Single ML detector (all sensors)")
    for i, f in enumerate([2, 4]):
        vv = v[((v.attacker == "targeted") | (v.attacker == "none")) & (v.f == f)].sort_values("k")
        ax.plot(vv.k, 100 * vv.false_trip, color=C[[0, 2][i]], lw=2, marker="o", ms=5,
                label=f"Resilient vote, trip needs > {f} sensors")
    ax.set_xlabel("Sensors controlled by the attacker (targeted)")
    ax.set_ylabel("Normal events that trip the breaker (%)")
    ax.set_title("Fake-fault (spoofing) attack: false trips", loc="left", color=INK, fontsize=11)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "fdia_false_trips.svg")


def feeder_map():
    from sim import I_SENSORS, V_SENSORS, compute_zones
    coords = pd.read_csv(os.path.join(ROOT, "feeder", "BusCoords.dat"), sep=r"[\s,]+", header=None,
                         names=["bus", "x", "y"], engine="python", dtype={"bus": str})
    coords["bus"] = coords.bus.str.lower()
    zones = compute_zones()
    import opendssdirect as dss
    fig, ax = plt.subplots(figsize=(8, 7))
    xy = coords.set_index("bus")[["x", "y"]]
    for name in dss.Circuit.AllElementNames():
        if not name.lower().startswith(("line.", "transformer.")):
            continue
        dss.Circuit.SetActiveElement(name)
        b = [s.split(".")[0].lower() for s in dss.CktElement.BusNames()]
        b = [k[:-1] if k not in xy.index and k.endswith("r") else k for k in b]  # regulator output buses
        if all(k in xy.index for k in b):
            ax.plot(xy.loc[b, "x"], xy.loc[b, "y"], color="#c9c8c2", lw=1.2, zorder=1)
    for a, bb in [("151", "300"), ("54", "94")]:  # normally-open tie switches Sw7, Sw8
        ax.plot(xy.loc[[a, bb], "x"], xy.loc[[a, bb], "y"], color=INK2, lw=1.4, ls="--", zorder=1)
    zc = {"Z1": C[0], "Z2": C[1], "Z3": C[2], "Z4": C[6], "Z5": C[4]}
    for z, col in zc.items():
        bs = [b for b, zz in zones.items() if zz == z and b in xy.index]
        ax.scatter(xy.loc[bs, "x"], xy.loc[bs, "y"], s=18, color=col, zorder=3, edgecolor="#fcfcfb", linewidth=0.6)
        cx, cy = xy.loc[bs, "x"].mean(), xy.loc[bs, "y"].mean()
        ax.text(cx, cy, z, fontsize=13, weight="bold", color=INK, zorder=5, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec=col, lw=1.5))
    vs = [b.lower() for b in V_SENSORS if b.lower() in xy.index]
    ax.scatter(xy.loc[vs, "x"], xy.loc[vs, "y"], s=90, facecolor="none", edgecolor=INK, lw=1.3, zorder=4, label="Voltage sensor (bus)")
    for el in I_SENSORS:
        dss.Circuit.SetActiveElement(f"Line.{el}")
        b = [s.split(".")[0].lower() for s in dss.CktElement.BusNames()]
        b = [k for k in b if k in xy.index]
        if b:
            p = xy.loc[b].mean()
            ax.scatter(p.x, p.y, marker="s", s=55, color=INK, zorder=4)
            ax.annotate(el, (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.scatter([], [], marker="s", s=55, color=INK, label="Current sensor (line)")
    ax.plot([], [], color=INK2, ls="--", label="Normally-open tie switch")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_title("IEEE 123-node feeder: protection zones and the 23 sensors", loc="left", color=INK, fontsize=11)
    save(fig, "feeder_map.svg")


if __name__ == "__main__":
    zone_gap()
    fdia()
    feeder_map()
