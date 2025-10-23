import torch

def triangle_consistency_loss(scores, triplets, margin=0.0):
    si = scores[triplets[:,0]]
    sj = scores[triplets[:,1]]
    sk = scores[triplets[:,2]]
    loss1 = torch.clamp(margin + sj - si, min=0)
    loss2 = torch.clamp(margin + sk - sj, min=0)
    loss3 = torch.clamp(margin + sk - si, min=0)
    return (loss1 + loss2 + 0.5*loss3).mean()
