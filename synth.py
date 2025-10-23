import argparse, numpy as np, pandas as pd

def generate(n=200, m=5000, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    # latent true scores
    s = rng.standard_normal(n)
    rows = []
    for _ in range(m):
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j: 
            continue
        # probability i beats j via logistic on score diff + noise
        p = 1.0 / (1.0 + np.exp(-(s[i] - s[j] + rng.normal(0, noise))))
        y = 1 if rng.random() < p else 0
        rows.append((i, j, y))
    df = pd.DataFrame(rows, columns=["source_id","target_id","label"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--m", type=int, default=5000)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/ncaa_sample.csv")
    args = ap.parse_args()
    df = generate(args.n, args.m, args.noise, args.seed)
    os = __import__("os")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} edges.")

if __name__ == "__main__":
    main()
