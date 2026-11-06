"""The D2PPI pair-scoring head (inference only).

A protein is a set of domain embeddings plus one whole-protein embedding, all from a frozen protein language model
(width D). The head scores a pair (A, B) from two symmetric branches:

  marginal pool    each protein's projected domains -> [mean; max; projected whole-protein vector] = v (768-d);
                   pair features [vA + vB, |vA - vB|, vA * vB]                                        (2304-d)
  domain-pair MIL  CompatMap: 8 low-rank bilinear maps score every domain pair (m x n per map); each map is read
                   out as max, mean, coverage and imbalance                                           (32-d)

The 2336-d concatenation goes through an MLP (LayerNorm -> 256 -> 128 -> 1). The five hidden Linear layers are
spectral-normalised. Swapping A and B leaves every feature unchanged, so f(A, B) = f(B, A).
"""

import torch
import torch.nn as nn

NEG = -1e4  # score of a padded domain pair


def masked_mean(x, m):
    """mean over the domain axis, ignoring padding. x: (B, T, d), m: (B, T) bool."""
    w = m.unsqueeze(-1).float()
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


def masked_max(x, m):
    """max over the domain axis, ignoring padding."""
    return x.masked_fill(~m.unsqueeze(-1), torch.finfo(x.dtype).min).max(1).values


class CompatMap(nn.Module):
    """Domain-pair MIL: `heads` low-rank bilinear compatibility maps over all m x n domain pairs.

    s_ij^(h) = <U_h a_i, U_h b_j> / sqrt(rank), read out per map as
    max (some domain pair fits), mean, coverage (rho_A + rho_B) / 2 and imbalance |rho_A - rho_B|,
    where rho_A = mean_i max_j s_ij (how many domains of A have a good partner in B).
    """

    def __init__(self, d, heads=8, rank=32):
        super().__init__()
        self.h, self.r = heads, rank
        self.proj = nn.Linear(d, heads * rank, bias=False)
        self.out_dim = heads * 4

    def forward(self, x1, m1, x2, m2):
        B, n1, _ = x1.shape
        n2 = x2.shape[1]
        p1 = self.proj(x1).view(B, n1, self.h, self.r)
        p2 = self.proj(x2).view(B, n2, self.h, self.r)
        s = torch.einsum("bihr,bjhr->bhij", p1, p2).float() / (self.r**0.5)
        valid = m1[:, None, :, None] & m2[:, None, None, :]
        sm = s.masked_fill(~valid, NEG)
        cnt = valid.sum((2, 3)).clamp(min=1).float()
        gmax = sm.amax((2, 3))
        gmean = (s * valid).sum((2, 3)) / cnt
        rmean = (sm.amax(3) * m1[:, None, :]).sum(2) / m1.sum(1).clamp(min=1)[:, None].float()
        cmean = (sm.amax(2) * m2[:, None, :]).sum(2) / m2.sum(1).clamp(min=1)[:, None].float()
        return torch.cat([gmax, gmean, 0.5 * (rmean + cmean), (rmean - cmean).abs()], dim=-1)


class D2PPI(nn.Module):
    """The released head. `in_dim` = encoder width (1280 for ESM-2-650M, 5120 for ESM-2-15B).

    The module layout (names and Sequential indices) is the one the checkpoints were trained with, so their state
    dicts load as they are.
    """

    def __init__(self, in_dim=1280, d=256, dropout=0.2, compat_heads=8, compat_rank=32):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))  # every domain
        self.gproj = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))  # whole-protein vector
        self.pair = CompatMap(d, compat_heads, compat_rank)
        head_in = 3 * (3 * d) + self.pair.out_dim  # 2304 + 32
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )
        for m in (self.proj[0], self.gproj[0], self.pair.proj, self.head[1], self.head[4]):
            nn.utils.parametrizations.spectral_norm(m)

    def forward(self, d1, m1, g1, d2, m2, g2):
        """d*: (B, T, in_dim) standardised domain embeddings, m*: (B, T) bool mask, g*: (B, in_dim) standardised
        whole-protein embeddings. Returns logits (B,); sigmoid gives P(interact)."""
        x1, x2 = self.proj(self.in_norm(d1)), self.proj(self.in_norm(d2))
        va = torch.cat([masked_mean(x1, m1), masked_max(x1, m1), self.gproj(self.in_norm(g1))], dim=-1)
        vb = torch.cat([masked_mean(x2, m2), masked_max(x2, m2), self.gproj(self.in_norm(g2))], dim=-1)
        z = torch.cat([va + vb, (va - vb).abs(), va * vb, self.pair(x1, m1, x2, m2)], dim=-1)
        return self.head(z).squeeze(-1)
