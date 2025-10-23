import torch
import numpy as np

class EdgeBatchSampler:
    def __init__(self, df, batch_edges=8192, seed=42):
        self.df = df.reset_index(drop=True)
        self.batch_edges = batch_edges
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = np.arange(len(self.df))
        self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_edges):
            sel = idx[start:start+self.batch_edges]
            yield self.df.iloc[sel].reset_index(drop=True)

class TripletSampler:
    def __init__(self, n, num_triplets=4096, seed=0):
        self.n = n
        self.num_triplets = num_triplets
        self.rng = np.random.default_rng(seed)

    def sample(self):
        i = self.rng.integers(0, self.n, size=self.num_triplets)
        j = self.rng.integers(0, self.n, size=self.num_triplets)
        k = self.rng.integers(0, self.n, size=self.num_triplets)
        return torch.tensor(np.stack([i,j,k], axis=1), dtype=torch.long)
