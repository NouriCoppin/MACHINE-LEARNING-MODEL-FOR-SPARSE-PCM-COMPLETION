import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseDirectedGNNLayer(nn.Module):
    """Directed message-passing using sparse SpMM (COO)."""
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, X, A_coo: torch.Tensor):
        AX = torch.sparse.mm(A_coo, X)  # SpMM
        H = self.lin(AX)
        H = self.dropout(F.relu(H))
        return H

class ProximalFiedlerRefinement(nn.Module):
    def __init__(self, iters=10, tau=0.5):
        super().__init__()
        self.iters = iters
        self.tau = tau

    def forward(self, scores, L_sym):
        v = scores
        for _ in range(self.iters):
            v = torch.sparse.mm(L_sym, v)
            v = v - self.tau * torch.sign(v)
            v = F.normalize(v, dim=0)
        return v

class GNNRank(nn.Module):
    def __init__(self, n_nodes, d=64, L=2, dropout=0.0, refine_iters=10, refine_tau=0.5):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, d)
        self.layers = nn.ModuleList([SparseDirectedGNNLayer(d, d, dropout) for _ in range(L)])
        self.scorer = nn.Linear(d, 1, bias=False)
        self.refine = ProximalFiedlerRefinement(iters=refine_iters, tau=refine_tau)

    def forward(self, A_coo, L_sym):
        X = self.embed.weight
        for gnn in self.layers:
            X = X + gnn(X, A_coo)
        s = self.scorer(X)
        s_ref = self.refine(s, L_sym)
        return s_ref.squeeze(-1)
