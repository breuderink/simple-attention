import torch
from torch import nn
from .shepards import ShepardsGatedAttentionBase


class PreNorm(nn.Module):
    def __init__(self, fn, norm) -> None:
        super().__init__()
        self.fn = fn
        self.norm = norm

    def forward(self, X, *args, **kwargs):
        return X + self.fn(self.norm(X), *args, **kwargs)


class ReZero(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, X, *args, **kwargs):
        return X + self.alpha * self.fn(X, *args, **kwargs)


def autoregressive_mask(n):
    position_query = torch.arange(n)[None, :, None]
    position_value = torch.arange(n)[None, None, :]
    return position_value > position_query



class Encoder(nn.Module):
    def __init__(self, *, dims_in, depth=1, prenorm=False, rezero=True, heads=8):
        super().__init__()
        steps = []

        for _ in range(depth):
            module = ShepardsGatedAttentionBase(dims_in=dims_in, heads=heads)
            if rezero:
                module = ReZero(module)
            if prenorm:
                norm = nn.LayerNorm(dims_in)
                module = PreNorm(module, norm)

            steps.append(module)

        self.steps = nn.ModuleList(steps)

    def forward(self, X, mask=None):
        for s in self.steps:
            X = s(X, mask=mask)

        return X