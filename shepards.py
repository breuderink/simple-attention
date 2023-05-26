# %%
import torch
from torch import nn
from einops import rearrange


# %%
def shepards_attention(query, key, value, mask=None, p=2, eps=1e-8):
    # Compute query-key distance.
    distance = torch.cdist(query, key)
    distance = torch.maximum(distance, torch.full_like(distance, eps))

    # Computer masked, inverse-distance weight.
    weight = torch.pow(distance, -p)
    if mask is not None:
        weight = weight.masked_fill(mask, 0)

    # Compute attention by normalizing.
    total = torch.sum(weight, dim=2, keepdims=True)
    total = torch.maximum(total, torch.full_like(total, eps))
    attention = weight / total

    # Apply attention.
    output = torch.bmm(attention, value)
    return output, attention


def shepards_MHA(Q, K, V, heads=8, **kwargs):
    Q = rearrange(Q, "b n (h d) -> (b h) n d", h=heads)
    K = rearrange(K, "b n (h d) -> (b h) n d", h=heads)
    V = rearrange(V, "b n (h d) -> (b h) n d", h=heads)

    Y, A = shepards_attention(Q, K, V, **kwargs)

    print(A.shape)
    Y = rearrange(Y, "(b h) n d -> b n (h d)", h=heads)
    A = rearrange(A, "(b h) n1 n2 -> b n1 (h n2)", h=heads)
    return Y, A


class ShepardsLayer(nn.Module):
    def __init__(self, *, d: int, d_qk: int, d_vg: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.splits = [d_qk, d_qk, d_vg, d_vg]

        self.to_qkvg = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d_qk + 2 * d_vg))
        self.to_out = nn.Linear(d_vg, d)

    def forward(self, X, mask=None):
        Q, K, V, G = torch.split(self.to_qkvg(X), self.splits, dim=2)
        A, _ = shepards_MHA(Q, K, V, mask=mask)
        return X + self.to_out(A * G)


def autoregressive_mask(n):
    position_query = torch.arange(n)[None, :, None]
    position_value = torch.arange(n)[None, None, :]
    return position_value >= position_query


# %%

import plotly.express as px

b, n, d = 2, 10, 32
X = torch.randn(b, n, d)

model = ShepardsLayer(d=d, d_qk=d, d_vg=2 * d, heads=4)
mask = autoregressive_mask(n)

Y1 = model(X, mask=mask)
X[:, n // 2, :] = 0
Y2 = model(X, mask=mask)

# TODO: test autoregressive mask.
torch.all(Y1 == Y2, dim=2)

# %%

# TODO: test that MHA is equivalent to parallel regular attention.
heads = 2

Q = torch.randn(b, n, d)
K = torch.randn(b, n, d)
V = torch.randn(b, n, 2 * d)

Y, A = shepards_attention(Q, K, V, mask=mask)
