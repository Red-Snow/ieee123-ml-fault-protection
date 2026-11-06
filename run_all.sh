#!/usr/bin/env bash
# Full pipeline: fetch feeder -> simulate -> benchmark -> topology-aware zone -> FDIA -> figures
# Extra arguments are passed through, e.g.  bash run_all.sh --quick
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m ieee123fp all "$@"
