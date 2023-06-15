import torch
from torch import nn
from typing import List


def shepards_MHA(Q, K, V, mask=None, p=2, eps=1e-4):
    W = torch.pow(eps + torch.cdist(Q, K), -p)
    if mask is not None:
        W = W.masked_fill(mask, 0)
    W = W / (eps + torch.sum(W, dim=-1, keepdims=True))
    return W @ V


class MultiHeadProjections(nn.Module):
    def __init__(self, *, heads: int, dim_in: int, dims_out: List[int]):
        super().__init__()
        self.heads = heads
        self.dims_out = dims_out

        self.project_in = nn.Linear(dim_in, heads * sum(dims_out))

    def forward(self, X):
        b, t, _ = X.shape
        h, dd = self.heads, self.dims_out

        P = self.project_in(X)  # (b, t, h * sum(dd))
        P = P.view(b, t, h, sum(dd))  # (b, t, h, sum(dd))
        P = P.permute((2, 0, 1, 3))  # (h, b, t, sum(dd))

        return torch.split(P, dd, dim=-1)  # (h, b, t, d_i)


class ShepardsGatedAttentionBase(nn.Module):
    def __init__(
        self, *, dims_in: int, heads: int, dims_att: int = None, dims_ff: int = None
    ):
        super().__init__()
        dims_att = dims_att if dims_att else 4
        dims_ff = dims_ff if dims_ff else dims_in // heads

        self.project_in = MultiHeadProjections(
            heads=heads, dim_in=dims_in, dims_out=[dims_att, dims_att, dims_ff, dims_ff]
        )
        self.project_out = nn.Linear(dims_ff * heads, dims_in)

    def forward(self, X, mask=None):
        b, t, _ = X.shape

        Q, K, V, G = self.project_in(X)
        h, _, _, e = V.shape

        A = shepards_MHA(Q, K, V, mask=mask)  # (h, b, t, d)
        H = G * A  # (h, b, t, d)
        H = H.permute((1, 2, 0, 3))  # (b, t, h, d)
        H = H.view(b, t, h * e)
        Y = self.project_out(H)
        return Y
