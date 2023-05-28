import torch
from torch import nn


def shepards_MHA(Q, K, V, mask=None, p=2, eps=1e-4):
    W = torch.pow(eps + torch.cdist(Q, K), -p)
    if mask is not None:
        W = W.masked_fill(mask, 0)
    W = W / (eps + torch.sum(W, dim=-1, keepdims=True))
    return W @ V


class ShepardsGatedAttention(nn.Module):
    def __init__(self, *, d: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.d = d

        self.project_in = nn.Sequential(
            nn.LayerNorm(d, elementwise_affine=False),
            nn.Linear(d, 4 * d),
        )
        self.project_out = nn.Linear(d, d)

    def QKVG(self, X):
        b, t, d = X.shape
        h = self.heads

        P = self.project_in(X)  # (b, t, 4 * d)
        P = P.view(b, t, h, 4 * d // h)  # (b, t, h, 4 * d / h)
        P = P.transpose(1, 2)  # (b, h, t, 4 * d / h)

        Q, K, V, G = torch.split(P, d // h, dim=-1)  # (b, h, t, _)
        return Q, K, V, G

    def forward(self, X, mask=None):
        b, t, d = X.shape

        Q, K, V, G = self.QKVG(X)
        A = shepards_MHA(Q, K, V, mask=mask)  # (b, h, t, d / h)
        O = (G * A).transpose(2, 1).view(b, t, d)  # (b, t, d)

        return X + self.project_out(O)


def autoregressive_mask(n):
    position_query = torch.arange(n)[None, :, None]
    position_value = torch.arange(n)[None, None, :]
    return position_value > position_query
