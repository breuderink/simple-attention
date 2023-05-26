# %%
import torch
from torch import nn
import matplotlib.pyplot as plt

# %%
def shepards_interpolation(Q, K, V, mask=None, p=2, eps=1e-8):
    D = torch.cdist(Q, K)
    W = torch.pow(D.clamp(min=eps), -p)
    if mask:
        W = torch.where(mask, W, torch.zeros(1))
    A = W / torch.sum(W, dim=2, keepdims=True)
    Y = torch.bmm(A, V)
    return Y


class ShepardsLayer(nn.Module):
    def __init__(self, *, d, d_qk, d_v):
        super().__init__()
        self.norm = nn.Identity()  # FIXME
        self.Q = nn.Linear(d, d_qk)
        self.K = nn.Linear(d, d_qk)
        self.V = nn.Linear(d, d_v)
        self.gate = nn.Linear(d, d_v)
        self.proj = nn.Linear(d_v, d)

    def forward(self, X):
        N = self.norm(X)
        A = shepards_interpolation(self.Q(N), self.K(N), self.V(N))
        G = self.gate(N)
        return X + self.proj(A * G)


# %%
K = torch.linspace(-10, 10, 16)[None, :, None]
V = torch.sin(K) / K
Q = torch.linspace(-20, 20, 500)[None, :, None]


plt.plot(K.flatten(), V.flatten(), "*", label="data")
plt.plot(
    Q.flatten(), shepards_interpolation(Q, K, V, p=2).flatten(), label="interpolation"
)
plt.xlabel("K")
plt.ylabel("V")
plt.legend()

# %%

dims = 64
X = torch.randn(1, 100, 3)

model = ShepardsLayer(d=3, d_qk=16, d_v=32)
Y = model(X)

# %%
