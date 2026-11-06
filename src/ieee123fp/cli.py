"""Command-line interface:  ieee123fp <command>  (or  python -m ieee123fp <command>).

    ieee123fp fetch                 download the IEEE 123-node feeder model
    ieee123fp generate              simulate 30,000 events (5 layouts x 6,000)
    ieee123fp benchmark             black-box models, random split vs leave-one-topology-out
    ieee123fp zone                  topology-aware zone localisation
    ieee123fp fdia                  false-data-injection study and defences
    ieee123fp figures               SVG figures from saved results
    ieee123fp all                   everything above, in order
    ieee123fp all --quick           a ~1 min smoke run on 1,000 events (writes under data/quick/)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Paths

QUICK_PER_TOPO = 200
STEPS = ["fetch", "generate", "benchmark", "zone", "fdia", "figures"]


def _paths(a: argparse.Namespace) -> Paths:
    base = Paths.quick() if a.quick else Paths.default()
    return Paths(Path(a.data) if a.data else base.data,
                 Path(a.results) if a.results else base.results,
                 Path(a.figures) if a.figures else base.figures)


def _step(name: str, a: argparse.Namespace, paths: Paths):
    log = logging.getLogger("ieee123fp")
    t0 = time.time()
    log.info("== %s ==", name)
    if name == "fetch":
        from .feeder import fetch
        fetch(force=a.force)
    elif name == "generate":
        from .dataset import generate
        per = a.per_topo or (QUICK_PER_TOPO if a.quick else 6000)
        generate(paths, per_topo=per, chunks=a.chunks, workers=a.workers)
    elif name == "benchmark":
        from .experiments import benchmark
        benchmark.run(paths)
    elif name == "zone":
        from .experiments import zone_loto
        zone_loto.run(paths)
    elif name == "fdia":
        from .experiments import fdia
        fdia.run(paths)
    elif name == "figures":
        from . import figures
        figures.run(paths)
    log.info("== %s done in %.0fs ==", name, time.time() - t0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ieee123fp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=STEPS + ["all"])
    p.add_argument("--quick", action="store_true",
                   help=f"smoke run: {QUICK_PER_TOPO} events per layout, outputs under data/quick/")
    p.add_argument("--data", help="raw phasor archive (.npz); default data/raw_phasors.npz")
    p.add_argument("--results", help="results directory; default results/")
    p.add_argument("--figures", help="figures directory; default figures/")
    g = p.add_argument_group("generate")
    g.add_argument("--per-topo", type=int, help="events per layout (default 6000)")
    g.add_argument("--chunks", type=int, default=6, help="seeded jobs per layout (changing it changes the data)")
    g.add_argument("--workers", type=int, help="worker processes (default: all cores; does not change the data)")
    p.add_argument("--force", action="store_true", help="fetch: re-download even if files are present")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")
    paths = _paths(a)
    for name in (STEPS if a.command == "all" else [a.command]):
        _step(name, a, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
