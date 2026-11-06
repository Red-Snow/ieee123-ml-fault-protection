"""Single source of truth for paths and study-wide constants.

Every other module imports from here, so changing a sensor list, a noise level or an
output directory happens in exactly one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
FEEDER_DIR = Path(os.environ.get("IEEE123FP_FEEDER_DIR", ROOT / "feeder"))


@dataclass(frozen=True)
class Paths:
    """Where one pipeline run reads and writes. `Paths.default()` reproduces the
    committed results; `Paths.quick()` keeps a smoke run away from them."""

    data: Path      # raw phasor archive (.npz); metadata sits next to it as *_meta.csv
    results: Path   # metrics (CSV/JSON)
    figures: Path   # SVG figures

    @property
    def meta(self) -> Path:
        return self.data.with_name(self.data.stem + "_meta.csv")

    @classmethod
    def default(cls) -> Paths:
        return cls(ROOT / "data" / "raw_phasors.npz", ROOT / "results", ROOT / "figures")

    @classmethod
    def quick(cls) -> Paths:
        q = ROOT / "data" / "quick"
        return cls(q / "raw_phasors.npz", q / "results", q / "figures")

    def ensure(self) -> Paths:
        for p in (self.data.parent, self.results, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------
# Measurement points: fixed hardware, the same sensors in every layout
# --------------------------------------------------------------------------
V_SENSORS = ["149", "13", "18", "35", "52", "60", "67", "76", "97", "101", "108", "300", "151", "450", "83"]
I_SENSORS = ["L115", "Sw2", "Sw3", "Sw4", "Sw5", "Sw7", "L108", "L67"]
SENSORS = [f"V{b}" for b in V_SENSORS] + [f"I{line}" for line in I_SENSORS]
SUBSTATION_SENSOR = "L115"   # always fed from the source: the polarising reference

# --------------------------------------------------------------------------
# Events and labels
# --------------------------------------------------------------------------
FAULT_TYPES = ["NORMAL", "SLG", "LL", "LLG", "LLL"]
TYPES = FAULT_TYPES[1:]
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
CLASS_WEIGHTS = {"NORMAL": 0.20, "SLG": 0.32, "LL": 0.16, "LLG": 0.16, "LLL": 0.16}
PHASES_NEEDED = {"SLG": 1, "LL": 2, "LLG": 2, "LLL": 3}
# fault resistance ranges (ohm), sampled log-uniformly
RF_RANGE = {"SLG": (0.05, 40.0), "LL": (0.01, 3.0), "LLG": (0.05, 20.0), "LLL": (0.01, 3.0)}

# --------------------------------------------------------------------------
# Grid / simulation
# --------------------------------------------------------------------------
SOURCE_MVASC3, SOURCE_MVASC1 = 200, 210   # ~28 kA three-phase at 4.16 kV
NOMINAL_ANG = np.array([0.0, -120.0, 120.0])
SUBSTATION_BUS = "150"
CAPACITOR = "C83"

# --------------------------------------------------------------------------
# Measurement noise (instrument transformer + PMU error, 1 sigma)
# --------------------------------------------------------------------------
NOISE_MAG_SD = 0.005
NOISE_ANG_SD_DEG = 0.2
NOISE_SEED = 0
