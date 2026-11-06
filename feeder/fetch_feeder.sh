#!/usr/bin/env bash
# Download the IEEE 123-node test feeder (EPRI OpenDSS model, BSD licence) at a pinned commit
# and verify SHA-256 checksums. URLs and checksums live in src/ieee123fp/feeder.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m ieee123fp fetch "$@"
