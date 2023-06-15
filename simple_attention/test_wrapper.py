from io import BytesIO

import torch
from torch import nn

import onnxruntime as ort

from .wrapper import (
    autoregressive_mask,
    PreNorm,
    ReZero,
    Encoder,
)


def test_autoregressive_mask():
    n = 5
    mask = autoregressive_mask(n)
    assert mask.shape == (1, n, n)

    for q in range(5):
        for k in range(5):
            assert mask[0, q, k] == (k > q)


def test_PreNorm():
    d = 128
    preNorm = PreNorm(nn.Identity(), nn.LayerNorm(d))

    assert sum(p.numel() for p in preNorm.parameters()) == 2 * d

    X = torch.randn(1, 2, d)
    N = preNorm.norm(X)
    Y = preNorm(N)

    torch.testing.assert_close(2 * N, Y)


def test_ReZero():
    reZero = ReZero(nn.Identity())

    assert sum(p.numel() for p in reZero.parameters()) == 1
    torch.testing.assert_close(reZero.alpha.data, torch.zeros(1))

    X = torch.randn(1, 2, 3)
    Y = reZero(X)
    torch.testing.assert_close(X, Y)

    b, n, d, h, dims_ff, dims_att = 1, 1, 16, 2, 32, 3
    block = Encoder(
        dims_in=d,
        depth=1,
        prenorm=False,
        rezero=True,
        heads=h,
        dims_ff=dims_ff,
        dims_att=dims_att,
    )

    X = torch.randn(b, n, d)
    Y = block(X)

    assert X.shape == Y.shape

    n_norm = 1
    n_proj_in = (d + 1) * h * (2 * dims_att + 2 * dims_ff)
    n_proj_out = (h * dims_ff + 1) * d
    assert sum([p.numel() for p in block.parameters()]) == sum(
        [n_norm, n_proj_in, n_proj_out]
    )


def test_ONNX_export():
    b, n, d = 2, 10, 32
    block = Encoder(dims_in=d)

    X = torch.randn(b, n, d)
    mask = torch.rand(b, n, n) < 0.5
    Y = block(X, mask)

    f = BytesIO()
    torch.onnx.export(
        block,
        (X, mask),
        f,
        input_names=["X", "mask"],
        output_names=["Y"],
        dynamic_axes={
            "X": {0: "batch", 1: "token"},
            "Y": {0: "token", 1: "token", 2: "token"},
        },
    )

    ort_session = ort.InferenceSession(f.getvalue())
    (out,) = ort_session.run(None, {"X": X.numpy(), "mask": mask.numpy()})
    Y2 = torch.from_numpy(out)
    torch.testing.assert_close(Y2, Y, atol=1e-3, rtol=1e-3)
