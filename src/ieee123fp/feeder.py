"""Download the IEEE 123-node test feeder (EPRI OpenDSS model, BSD licence) at a pinned
commit of the DSS-Extensions test-case repository and verify SHA-256 checksums."""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from .config import FEEDER_DIR

log = logging.getLogger(__name__)

COMMIT = "3b208397160213cae4a9e2d0a7d1aa3528ce26e1"
BASE = f"https://raw.githubusercontent.com/dss-extensions/electricdss-tst/{COMMIT}"
CASE = "Version8/Distrib/IEEETestCases"
FILES = {  # local name: (path in repo, sha256)
    "IEEE123Master.dss": (f"{CASE}/123Bus/IEEE123Master.dss",
                          "c92a69d9b218b1b2646ec7911783826229309038e72f16b848304c0457c0a54d"),
    "IEEE123Regulators.DSS": (f"{CASE}/123Bus/IEEE123Regulators.DSS",
                              "cd6ad77980dc018631ce39faab528afce40b0b9a50a6bc1c9ac8049ffb8207d7"),
    "IEEE123Loads.DSS": (f"{CASE}/123Bus/IEEE123Loads.DSS",
                         "fd28f52836205506a1980b52c5a7e12ca585aa626c52a6249502c1c3b2fc2255"),
    "BusCoords.dat": (f"{CASE}/123Bus/BusCoords.dat",
                      "0b409c0e5d99f365c0140bf38280c1d64932032f1ecb8ef83ae54d4ed0d2716f"),
    "IEEELineCodes.DSS": (f"{CASE}/IEEELineCodes.DSS",
                          "71bc16ef9c6084b9e3bab74fbdf967080b79624683465e1483b787ff160f58b3"),
    "LICENSE-OpenDSS.txt": ("License.txt",
                            "f13543277f32a6437d4f583200858f0988d3acc9598a6a1a292d710009ec21ef"),
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(dest: Path = FEEDER_DIR, force: bool = False) -> Path:
    """Download any missing (or corrupted) feeder file into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    for name, (rel, digest) in FILES.items():
        p = dest / name
        if p.exists() and not force and _sha256(p) == digest:
            log.info("%-22s ok (cached)", name)
            continue
        with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as r:
            p.write_bytes(r.read())
        if _sha256(p) != digest:
            p.unlink()
            raise RuntimeError(f"checksum mismatch for {name}")
        log.info("%-22s ok", name)
    log.info("IEEE 123-node feeder files ready in %s", dest)
    return dest
