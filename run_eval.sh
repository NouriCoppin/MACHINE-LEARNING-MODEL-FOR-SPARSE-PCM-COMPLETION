#!/usr/bin/env bash
set -euo pipefail
python -m src.eval --ckpt runs/exp1/best.pt --data.path data/ncaa_sample.csv
