"""Models.

Proposed: GLCM-conditioned patch-graph attention network (PGAT-FiLM)
  1. ResNet-50 backbone (ImageNet pretrained) -> feature map F in R^{C x H' x W'} (7 x 7 for 224 x 224 input).
  2. Each spatial cell of F is a graph node: x_i = W_p f_i  (1x1 projection to d dims), N = H' * W' nodes.
  3. Edges: 8-neighbour spatial grid  UNION  k nearest neighbours in feature space (cosine similarity),
     plus self-loops.  (Adjacency is recomputed for every image.)
  4. Feature-wise affine modulation (FiLM) of node features by the 14-D GLCM texture vector t:
         x_i <- (1 + gamma(t)) * x_i + beta(t)
  5. L multi-head graph attention layers (GAT):
         e_ij^h = LeakyReLU(a_h^T [W_h x_i || W_h x_j]),  alpha_ij^h = softmax_{j in N(i)} e_ij^h
         x_i'   = LayerNorm(x_i + ELU(||_h sum_{j in N(i)} alpha_ij^h W_h x_j))
  6. Readout: [mean_i x_i || max_i x_i || GAP(F) projected] -> dropout -> linear classifier.

Ablations: --no_graph (backbone + head), --no_film, --graph grid|knn|both.
Baselines: any timm model name, fine-tuned end to end with the same data, split and schedule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from common import GLCM_DIM


def grid_adjacency(h, w, device):
    idx = torch.arange(h * w, device=device).view(h, w)
    A = torch.eye(h * w, dtype=torch.bool, device=device)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), h + min(0, dy)); yd = slice(max(0, -dy), h + min(0, -dy))
            xs = slice(max(0, dx), w + min(0, dx)); xd = slice(max(0, -dx), w + min(0, -dx))
            A[idx[yd, xd].reshape(-1), idx[ys, xs].reshape(-1)] = True
    return A


def knn_adjacency(x, k):
    """x: (B, N, d) -> boolean (B, N, N), symmetric, with self-loops."""
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.transpose(1, 2)
    k = min(k + 1, x.shape[1])
    nn_idx = sim.topk(k, dim=-1).indices
    A = torch.zeros_like(sim, dtype=torch.bool).scatter_(-1, nn_idx, True)
    return A | A.transpose(1, 2)


class GATLayer(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout); self.norm = nn.LayerNorm(d)
        self.last_attention = None

    def forward(self, x, A):  # x (B,N,d), A (B,N,N) bool
        B, N, _ = x.shape
        Wx = self.W(x).view(B, N, self.h, self.dh)                      # B N H dh
        e = (Wx * self.a_dst).sum(-1).unsqueeze(2) + (Wx * self.a_src).sum(-1).unsqueeze(1)  # B N N H
        e = F.leaky_relu(e, 0.2).masked_fill(~A.unsqueeze(-1), float("-inf"))
        alpha = self.drop(torch.softmax(e, dim=2))                       # over neighbours j
        self.last_attention = alpha.detach()
        out = torch.einsum("bijh,bjhd->bihd", alpha, Wx).reshape(B, N, -1)
        return self.norm(x + F.elu(out))


class FiLM(nn.Module):
    def __init__(self, cond_dim, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # starts as identity

    def forward(self, x, t):
        gamma, beta = self.net(t).chunk(2, dim=-1)
        return (1 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class PGATFiLM(nn.Module):
    def __init__(self, n_classes, backbone="resnet50", pretrained=True, d=256, heads=4, layers=2, k=8,
                 graph="both", use_graph=True, use_film=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        c = self.backbone.num_features
        self.proj = nn.Conv2d(c, d, 1)
        self.gap_proj = nn.Linear(c, d)
        self.use_graph, self.use_film, self.graph, self.k = use_graph, use_film, graph, k
        self.film = FiLM(GLCM_DIM, d) if use_film else None
        self.gat = nn.ModuleList([GATLayer(d, heads, 0.1) for _ in range(layers)]) if use_graph else None
        head_in = (3 * d) if use_graph else (2 * d if use_film else d)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(head_in, n_classes)

    def adjacency(self, x, h, w):
        A = None
        if self.graph in ("grid", "both"):
            A = grid_adjacency(h, w, x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.graph in ("knn", "both"):
            Ak = knn_adjacency(x, self.k)
            A = Ak if A is None else (A | Ak)
        return A

    def forward(self, img, glcm):
        fmap = self.backbone.forward_features(img)                 # B C H W
        gap = self.gap_proj(fmap.mean((2, 3)))                     # B d
        if not self.use_graph:
            z = gap
            if self.use_film:                                      # texture still used, via FiLM on pooled feature
                z = torch.cat([gap, self.film(gap.unsqueeze(1), glcm).squeeze(1)], -1)
            return self.head(self.drop(z))
        x = self.proj(fmap)
        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)                           # B N d
        if self.use_film:
            x = self.film(x, glcm)
        A = self.adjacency(x, h, w)
        for layer in self.gat:
            x = layer(x, A)
        z = torch.cat([x.mean(1), x.max(1).values, gap], -1)
        return self.head(self.drop(z))


class TimmBaseline(nn.Module):
    def __init__(self, name, n_classes, pretrained=True, img_size=224):
        super().__init__()
        kw = {"img_size": img_size} if any(s in name for s in ("vit", "deit", "swin")) else {}
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=n_classes, **kw)

    def forward(self, img, glcm=None):
        return self.net(img)


ABLATIONS = {
    "proposed": dict(),
    "abl_backbone_only": dict(use_graph=False, use_film=False),
    "abl_no_film": dict(use_film=False),
    "abl_no_graph": dict(use_graph=False),
    "abl_grid_only": dict(graph="grid"),
    "abl_knn_only": dict(graph="knn"),
}


def build_model(name, n_classes, pretrained=True, img_size=224, **hp):
    if name in ABLATIONS:
        cfg = {**hp, **ABLATIONS[name]}
        return PGATFiLM(n_classes, pretrained=pretrained, **cfg)
    return TimmBaseline(name, n_classes, pretrained, img_size)


def uses_glcm(name):
    return name in ABLATIONS and ABLATIONS[name].get("use_film", True)
