import torch
from torch import nn


def shepards_MHA(Q, K, V, mask=None, p=2, eps=1e-4):
    W = torch.pow(eps + torch.cdist(Q, K), -p)
    if mask is not None:
        W = W.masked_fill(mask, 0)
    W = W / (eps + torch.sum(W, dim=-1, keepdims=True))
    return W @ V


def autoregressive_mask(n):
    position_query = torch.arange(n)[None, :, None]
    position_value = torch.arange(n)[None, None, :]
    return position_value > position_query


class ReZero(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, X, *args, **kwargs):
        return X + self.alpha * self.fn(X, *args, **kwargs)


class MultiHeadProjections(nn.Module):
    def __init__(self, *, heads: int, dim_in: int, dims_out: list[int]):
        super().__init__()
        self.heads = heads
        self.dims_out = dims_out

        self.project_in = nn.Linear(dim_in, heads * sum(dims_out))

    def forward(self, X):
        b, t, _ = X.shape
        h, dd = self.heads, self.dims_out

        P = self.project_in(X)  # (b, t, h * sum(dd))
        P = P.view(b, t, h, sum(dd))  # (b, t, h, sum(dd))
        P = P.transpose(1, 2)  # (b, h, t, sum(dd))

        return torch.split(P, dd, dim=-1)  # (b, h, t, d_i)


class ShepardsGatedAttentionBase(nn.Module):
    def __init__(self, *, dims_in: int, heads: int, dims_per_head: int = None):
        super().__init__()
        dims_per_head = dims_per_head if dims_per_head else dims_in // heads

        self.project_in = MultiHeadProjections(
            heads=heads, dim_in=dims_in, dims_out=4 * [dims_per_head]
        )
        self.project_out = nn.Linear(dims_per_head * heads, dims_in)

    def forward(self, X, mask=None):
        b, t, d = X.shape
        Q, K, V, G = self.project_in(X)
        A = shepards_MHA(Q, K, V, mask=mask)  # (b, h, t, d)
        H = (G * A).transpose(2, 1).view(b, t, d)  # (b, t, d)
        Y = self.project_out(H)
        return Y


class ShepardsGatedAttention(ReZero):
    def __init__(self, dims_in: int, *, heads: int = 8) -> None:
        fn = ShepardsGatedAttentionBase(dims_in=dims_in, heads=heads)
        super().__init__(fn)
