import argparse, os, yaml, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .utils.data import load_pairs_csv, split_edges
from .utils.graph import build_sparse_graph, build_laplacian_surrogate
from .utils.sampler import EdgeBatchSampler, TripletSampler
from .models.gnnrank import GNNRank
from .losses.triangle import triangle_consistency_loss

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data.path', dest='data_path', type=str, required=False)
    ap.add_argument('--outdir', type=str, default='runs/exp')
    return ap.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate_upset(model, edges_df, n, device):
    model.eval()
    A = build_sparse_graph(n, edges_df).to(device)
    L = build_laplacian_surrogate(A).to(device)
    scores = model(A, L)
    src = torch.tensor(edges_df['source_id'].to_numpy(), device=device)
    tgt = torch.tensor(edges_df['target_id'].to_numpy(), device=device)
    lbl = torch.tensor(edges_df['label'].to_numpy(), device=device)  # 0/1
    sdiff = scores[src] - scores[tgt]
    preds = (sdiff > 0).long()
    upset = (preds != lbl).float().mean().item()
    return upset

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_path:
        cfg['data']['path'] = args.data_path
    outdir = args.outdir or cfg.get('log',{}).get('outdir','runs/exp')
    os.makedirs(outdir, exist_ok=True)

    set_seed(cfg.get('seed', 42))

    # Data
    df, n = load_pairs_csv(cfg['data']['path'])
    if cfg['data'].get('num_nodes'):
        n = cfg['data']['num_nodes']
    df_tr, df_va, df_te = split_edges(df, cfg['data']['val_frac'], cfg['data']['test_frac'], cfg.get('seed',42))

    # Model
    model = GNNRank(
        n_nodes=n,
        d=cfg['model']['d'],
        L=cfg['model']['L'],
        dropout=cfg['model'].get('dropout', 0.0),
        refine_iters=cfg['refine'].get('iters', 10),
        refine_tau=cfg['refine'].get('tau', 0.5)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    clip = cfg['train'].get('grad_clip', 1.0)

    # Samplers
    sampler = EdgeBatchSampler(df_tr, batch_edges=cfg['train']['batch_edges'], seed=cfg.get('seed', 42))
    trip_sampler = TripletSampler(n, num_triplets=max(2048, cfg['train']['batch_edges']//2), seed=cfg.get('seed', 42))

    writer = SummaryWriter(log_dir=outdir)

    global_step = 0
    best_val = float('inf')
    best_ckpt = os.path.join(outdir, 'best.pt')

    for epoch in range(cfg['train']['epochs']):
        model.train()
        for batch_df in sampler:
            opt.zero_grad()

            # Build batch-induced sparse graph and laplacian surrogate
            A = build_sparse_graph(n, batch_df).to(device)
            L = build_laplacian_surrogate(A).to(device)

            scores = model(A, L)  # (n,)

            # Pairwise logistic loss
            src = torch.tensor(batch_df['source_id'].to_numpy(), device=device)
            tgt = torch.tensor(batch_df['target_id'].to_numpy(), device=device)
            lbl = torch.tensor(batch_df['label'].to_numpy(), dtype=torch.float32, device=device)
            sdiff = scores[src] - scores[tgt]
            pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(sdiff, lbl)

            # Triangle consistency
            triplets = trip_sampler.sample().to(device)
            tri_loss = triangle_consistency_loss(scores, triplets, margin=0.0)

            loss = pair_loss + cfg['loss']['lambda_delta'] * tri_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            if global_step % 10 == 0:
                writer.add_scalar('train/loss', float(loss.detach().cpu()), global_step)
                writer.add_scalar('train/pair_loss', float(pair_loss.detach().cpu()), global_step)
                writer.add_scalar('train/tri_loss', float(tri_loss.detach().cpu()), global_step)
            global_step += 1

        # Validation upset rate
        val_upset = evaluate_upset(model, df_va, n, device)
        writer.add_scalar('val/upset', val_upset, epoch)

        if val_upset < best_val:
            best_val = val_upset
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, best_ckpt)

        print(f"Epoch {epoch}: val_upset={val_upset:.4f}")

    writer.close()

if __name__ == '__main__':
    main()
