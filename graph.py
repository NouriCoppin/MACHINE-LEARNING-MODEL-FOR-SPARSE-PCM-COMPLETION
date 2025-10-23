import torch

def build_sparse_graph(n, edges_df):
    winners = edges_df[edges_df.label==1][["source_id","target_id"]].to_numpy()
    losers  = edges_df[edges_df.label==0][["target_id","source_id"]].to_numpy()
    pairs = None
    if len(winners) + len(losers) > 0:
        import numpy as np
        pairs = torch.tensor(np.vstack([winners, losers]), dtype=torch.long)
        src = pairs[:,0]; dst = pairs[:,1]
        indices = torch.stack([src, dst]).long()
        values = torch.ones(src.size(0), dtype=torch.float32)
    else:
        indices = torch.zeros((2,0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    return A

def build_laplacian_surrogate(A):
    A_sym = 0.5*(A + A.transpose(0,1))
    deg = torch.sparse.sum(A_sym, dim=1).to_dense()
    idx = torch.arange(len(deg))
    D = torch.sparse_coo_tensor(torch.stack([idx, idx]), deg, (len(deg), len(deg)))
    L = (D - A_sym).coalesce()
    return L
