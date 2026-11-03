#!/usr/bin/env bash
# Full pipeline: simulate -> train -> topology-aware model -> attack study -> figures
set -euo pipefail
cd "$(dirname "$0")"
bash feeder/fetch_feeder.sh
python src/generate.py --per-topo 6000
python src/train.py
python src/topo_aware.py
python src/attack.py
python src/figures.py
