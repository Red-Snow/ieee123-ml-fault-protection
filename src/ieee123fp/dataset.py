"""Generate the simulated dataset in parallel, and load it back as noisy features + labels."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NOISE_SEED, Paths
from .features import PHASOR_KEYS, add_noise, build_from
from .grid import TOPOLOGIES
from .simulation import Simulator, sample_class

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _work(job: tuple[str, int, int]):
    topo, n, seed = job
    sim = Simulator(seed)
    rows, arrs = [], {k: [] for k in PHASOR_KEYS}
    while len(rows) < n:
        s = sim.sample(topo, sample_class(sim.rng))
        if s is None:   # non-converged power flow: draw again
            continue
        rows.append(s.meta())
        for k in PHASOR_KEYS:
            arrs[k].append(getattr(s, k))
    return rows, {k: np.array(v) for k, v in arrs.items()}


def make_jobs(per_topo: int, chunks: int) -> list[tuple[str, int, int]]:
    """Split each layout's samples into `chunks` independent, seeded jobs.

    Seeds depend only on (layout index, chunk index), so the dataset is identical for any
    number of worker processes."""
    jobs = []
    for ti, t in enumerate(TOPOLOGIES):
        sizes = [per_topo // chunks + (c < per_topo % chunks) for c in range(chunks)]
        jobs += [(t, n, 1000 * ti + c) for c, n in enumerate(sizes) if n > 0]
    return jobs


def generate(paths: Paths, per_topo: int = 6000, chunks: int = 6, workers: int | None = None) -> pd.DataFrame:
    """Simulate `per_topo` events in each of the five layouts and save raw phasors + metadata."""
    jobs = make_jobs(per_topo, chunks)
    t0 = time.time()
    res = []
    with Pool(workers or os.cpu_count()) as p:
        for k, r in enumerate(p.imap(_work, jobs), 1):
            res.append(r)
            log.info("[%d/%d] chunks done, %.0fs", k, len(jobs), time.time() - t0)
    meta = pd.DataFrame([row for r in res for row in r[0]])
    paths.data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.data, **{k: np.concatenate([r[1][k] for r in res]) for k in PHASOR_KEYS})
    meta.to_csv(paths.meta, index=False)
    log.info("class counts per layout:\n%s", meta.groupby(["topo", "ftype"]).size().unstack())
    log.info("saved %d samples to %s in %.0fs", len(meta), paths.data, time.time() - t0)
    return meta


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    """X: (n, 208) features; meta: one row per event; ph: noisy phasors (n, sensors, 3)."""

    X: np.ndarray
    meta: pd.DataFrame
    ph: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.meta)

    def __iter__(self):   # allows  X, meta, ph = load(...)
        return iter((self.X, self.meta, self.ph))

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(self.X[mask], self.meta[mask].reset_index(drop=True), {k: v[mask] for k, v in self.ph.items()})

    def faults(self) -> Dataset:
        return self.subset(self.meta.is_fault.values == 1)


def load(path: Path | str | None = None, seed: int = NOISE_SEED) -> Dataset:
    """Load raw phasors, add measurement noise (fixed seed) and build features."""
    path = Path(path) if path else Paths.default().data
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Simulate it first:  ieee123fp generate")
    raw = np.load(path)
    meta = pd.read_csv(path.with_name(path.stem + "_meta.csv"))
    rng = np.random.default_rng(seed)
    ph = {k: add_noise(raw[k], rng) for k in PHASOR_KEYS}
    meta["is_fault"] = (meta.ftype != "NORMAL").astype(int)
    return Dataset(build_from(ph), meta, ph)
