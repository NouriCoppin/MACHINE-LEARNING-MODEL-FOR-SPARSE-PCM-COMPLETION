import argparse, torch, pandas as pd
from .utils.data import load_pairs_csv
from .utils.graph import build_sparse_graph, build_laplacian_surrogate
from .models.gnnrank import GNNRank

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data.path', dest='data_path', type=str, required=True)
    args = ap.parse_args()

    df, n = load_pairs_csv(args.data_path)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    model = GNNRank(n_nodes=n,
                    d=cfg['model']['d'], L=cfg['model']['L'],
                    dropout=cfg['model'].get('dropout',0.0),
                    refine_iters=cfg['refine'].get('iters',10),
                    refine_tau=cfg['refine'].get('tau',0.5))
    model.load_state_dict(ckpt['model_state'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    A = build_sparse_graph(n, df).to(device)
    L = build_laplacian_surrogate(A).to(device)

    with torch.no_grad():
        scores = model(A, L)

    out = pd.DataFrame({'node_id': list(range(n)), 'score': scores.cpu().numpy()})
    out.to_csv('scores.csv', index=False)
    print('Saved scores.csv')

if __name__ == '__main__':
    main()
