"""Make the README figures (SVG) from the saved result files. No retraining needed."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import opendssdirect as dss  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FEEDER_DIR, I_SENSORS, V_SENSORS, Paths  # noqa: E402
from .grid import compile_feeder, compute_zones, element_buses  # noqa: E402
from .models.baselines import MODEL_LABELS  # noqa: E402

log = logging.getLogger(__name__)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "svg.fonttype": "none", "svg.hashsalt": "ieee123", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
TOPO_LABEL = {"T0_base": "T0\nbase", "T1_sw7_sw5": "T1\nSw7 in, Sw5 out", "T2_sw7_sw3": "T2\nSw7 in, Sw3 out",
              "T3_loop": "T3\nlooped", "T4_sw8_l93": "T4\nSw8 in, L93 out"}


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    # round coordinates to 0.1 px: same picture, smaller files, stable diffs
    head, body = path.read_text().split("</metadata>", 1)
    path.write_text(head + "</metadata>" + re.sub(r"(-?\d+\.\d)\d+", r"\1", body))
    log.info("wrote %s", path)


def zone_gap(paths: Paths):
    m = pd.read_csv(paths.results / "metrics.csv")
    ta = pd.read_csv(paths.results / "metrics_topology_aware.csv")
    z = m[(m.task == "zone") & (m.protocol == "loto")]
    rand = m[(m.task == "zone") & (m.protocol == "random")].acc.max()
    topos = list(TOPO_LABEL)
    series = [(lab, z[z.model == m]) for m, lab in MODEL_LABELS.items()] + [("Topology-aware (this repo)", ta)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    x = np.arange(len(topos))
    for i, (lab, d) in enumerate(series):
        vals = [100 * d[d.held_out == t].acc.values[0] for t in topos]
        ax.bar(x + (i - 1.5) * w, vals, w - 0.03, color=C[i], label=lab, edgecolor="#fcfcfb", linewidth=0)
        if i == 3:
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 1.5) * w, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(100 * rand, color=INK2, lw=1, ls="--", label=f"Best model, random 80/20 split with layouts mixed ({100*rand:.1f}%)")
    ax.set_xticks(x, [TOPO_LABEL[t] for t in topos], fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Fault-zone accuracy on the unseen layout (%)")
    ax.set_title("Zone localisation when the grid layout was never seen in training", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    save(fig, paths.figures / "zone_generalisation.svg")


def fdia(paths: Paths):
    a = pd.read_csv(paths.results / "fdia.csv")
    v = pd.read_csv(paths.results / "fdia_resilient_vote.csv")
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
    save(fig, paths.figures / "fdia_false_trips.svg")


def feeder_map(paths: Paths):
    """Hand-written compact SVG (the matplotlib version is ~50 KB)."""
    coords = pd.read_csv(FEEDER_DIR / "BusCoords.dat", sep=r"[\s,]+", header=None,
                         names=["bus", "x", "y"], engine="python", dtype={"bus": str})
    coords["bus"] = coords.bus.str.lower()
    zones = compute_zones()
    compile_feeder("T0_base")
    xy = coords.set_index("bus")[["x", "y"]]
    x0, x1, y0, y1 = xy.x.min(), xy.x.max(), xy.y.min(), xy.y.max()
    W, PAD, TOP = 620, 20, 40
    k = (W - 2 * PAD) / (x1 - x0)
    H = int(TOP + (y1 - y0) * k + 2 * PAD + 30)
    P = lambda b: (round(PAD + (xy.loc[b, "x"] - x0) * k, 1), round(TOP + (y1 - xy.loc[b, "y"]) * k, 1))
    segs = []
    for name in dss.Circuit.AllElementNames():
        if not name.lower().startswith(("line.", "transformer.")):
            continue
        b = element_buses(name)
        b = [q[:-1] if q not in xy.index and q.endswith("r") else q for q in b]
        if len(b) == 2 and all(q in xy.index for q in b) and b[0] != b[1]:
            (ax_, ay), (bx, by) = P(b[0]), P(b[1])
            segs.append(f"M{ax_} {ay}L{bx} {by}")
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="DejaVu Sans, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
         f'<text x="{PAD}" y="22" font-size="14" fill="{INK}">IEEE 123-node feeder: protection zones and the 23 sensors</text>',
         f'<path d="{"".join(segs)}" stroke="#c9c8c2" stroke-width="1.5" fill="none"/>']
    tie = "".join(f"M{P(a)[0]} {P(a)[1]}L{P(b)[0]} {P(b)[1]}" for a, b in [("151", "300"), ("54", "94")])
    o.append(f'<path d="{tie}" stroke="{INK2}" stroke-width="1.6" stroke-dasharray="5 3" fill="none"/>')
    zc = {"Z1": C[0], "Z2": C[1], "Z3": C[2], "Z4": C[6], "Z5": C[4]}
    for z, col in zc.items():
        bs = sorted(b for b, zz in zones.items() if zz == z and b in xy.index)
        o.append(f'<g fill="{col}" stroke="#fcfcfb" stroke-width="0.8">'
                 + "".join(f'<circle cx="{P(b)[0]}" cy="{P(b)[1]}" r="3"/>' for b in bs) + "</g>")
    o.append(f'<g fill="none" stroke="{INK}" stroke-width="1.5">'
             + "".join(f'<circle cx="{P(b.lower())[0]}" cy="{P(b.lower())[1]}" r="7"/>' for b in V_SENSORS) + "</g>")
    for el in I_SENSORS:
        b = [q for q in element_buses(f"Line.{el}") if q in xy.index]
        px = round(np.mean([P(q)[0] for q in b]), 1)
        py = round(np.mean([P(q)[1] for q in b]), 1)
        o.append(f'<rect x="{round(px - 5, 1)}" y="{round(py - 5, 1)}" width="10" height="10" fill="{INK}"/>'
                 f'<text x="{round(px + 7, 1)}" y="{round(py - 6, 1)}" font-size="10" fill="{INK}">{el}</text>')
    for z, col in zc.items():
        bs = sorted(b for b, zz in zones.items() if zz == z and b in xy.index)
        cx = round(np.mean([P(b)[0] for b in bs]), 1)
        cy = round(np.mean([P(b)[1] for b in bs]), 1)
        o.append(f'<rect x="{round(cx - 17, 1)}" y="{round(cy - 13, 1)}" width="34" height="24" rx="4" fill="#fcfcfb" stroke="{col}" stroke-width="2"/>'
                 f'<text x="{cx}" y="{round(cy + 5, 1)}" font-size="15" font-weight="bold" text-anchor="middle" fill="{INK}">{z}</text>')
    ly = H - 18
    o.append(f'<circle cx="{PAD + 7}" cy="{ly}" r="7" fill="none" stroke="{INK}" stroke-width="1.5"/>'
             f'<text x="{PAD + 20}" y="{ly + 4}" font-size="11" fill="{INK}">Voltage sensor (bus)</text>'
             f'<rect x="{PAD + 160}" y="{ly - 5}" width="10" height="10" fill="{INK}"/>'
             f'<text x="{PAD + 176}" y="{ly + 4}" font-size="11" fill="{INK}">Current sensor (line)</text>'
             f'<path d="M{PAD + 320} {ly}h22" stroke="{INK2}" stroke-width="1.6" stroke-dasharray="5 3"/>'
             f'<text x="{PAD + 348}" y="{ly + 4}" font-size="11" fill="{INK}">Normally-open tie switch</text>')
    o.append("</svg>\n")
    out = paths.figures / "feeder_map.svg"
    out.write_text("\n".join(o))
    log.info("wrote %s", out)


def run(paths: Paths):
    paths.ensure()
    zone_gap(paths)
    fdia(paths)
    feeder_map(paths)
