#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config configs/default.yaml --data.path data/ncaa_sample.csv --outdir runs/exp1
