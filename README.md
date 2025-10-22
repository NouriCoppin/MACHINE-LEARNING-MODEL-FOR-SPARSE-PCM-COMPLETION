# MACHINE-LEARNING-MODEL-FOR-SPARSE-PCM-COMPLETION
introduce a new machine learning model for sparse pairwise comparison matrices (PCMs), combining classical PCM approaches with graph-based learning techniques. Numerical results are provided to demonstrate the effectiveness and scalability of the proposed method.

# GNNRank-Style Ranking with Consistency Loss (L_Δ)

This library trains a global ranking from pairwise comparisons using a directed GNN and a proximal Fiedler-refinement step.
We add an explicit **triangle (cycle) consistency loss** (\(\mathcal{L}_\Delta\)) and train efficiently via mini-batch
subgraph induction and sparse SpMM.

## Quickstart (once unzipped)
```bash
pip install -r requirements.txt
python -m src.utils.synth --n 200 --m 5000 --out data/ncaa_sample.csv
python -m src.train --config configs/default.yaml --data.path data/ncaa_sample.csv --outdir runs/exp1
python -m src.eval --ckpt runs/exp1/best.pt --data.path data/ncaa_sample.csv
```

Artifacts:
- `runs/exp1/` contains TensorBoard logs and `best.pt`
- `scores.csv` written by `src/eval.py`
