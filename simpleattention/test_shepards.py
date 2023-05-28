from io import BytesIO
import torch
import onnxruntime as ort
from simpleattention.shepards import autoregressive_mask, shepards_MHA, ShepardsGatedAttention


def test_autoregressive_mask():
    n = 5
    mask = autoregressive_mask(n)
    assert mask.shape == (1, n, n)

    for q in range(5):
        for k in range(5):
            assert mask[0, q, k] == (k > q)


def test_attention():
    b, n, d_qk, d_v = 2, 10, 8, 16

    K = torch.randn(b, n, d_qk)
    V = torch.randn(b, n, d_v)

    order = torch.randperm(n)
    Q, T = K[:, order], V[:, order]

    Y = shepards_MHA(Q, K, V)
    T = V[:, order]
    torch.testing.assert_close(Y, T)


def test_multi_head_attention():
    b, h, n, d_qk, d_v = 2, 4, 10, 8, 16

    Q = torch.randn(b, h, n, d_qk)
    K = torch.randn(b, h, n, d_qk)
    V = torch.randn(b, h, n, d_v)

    Y = shepards_MHA(Q, K, V)

    # Test that instances and heads are independent.
    for batch in range(b):
        for head in range(h):
            Y2 = shepards_MHA(Q[batch, head], K[batch, head], V[batch, head])
            torch.testing.assert_close(Y2, Y[batch, head])


def test_masked_attention():
    b, n_q, n_kv, d = 1, 3, 5, 16

    Q = torch.randn(b, n_q, d)
    K = torch.randn(b, n_kv, d)
    V = torch.randn(b, n_kv, d)

    mask = torch.rand(n_q, n_kv) < 0.5
    print(mask)

    Y = shepards_MHA(Q, K, V, mask=mask)

    for i in range(n_q):
        # Modify a single token.
        V2 = V.clone()
        V2[:, i] = 0

        # Recompute attention.
        Y2 = shepards_MHA(Q, K, V2, mask=mask)

        # Test correspondence between mask and unchanged values.
        unaffected = torch.all(torch.isclose(Y2, Y), dim=2).flatten()
        assert (unaffected == mask[:, i]).all()


def test_ShepardsGatedAttention():
    b, n, d = 1, 1, 16
    block = ShepardsGatedAttention(d=d)

    X = torch.randn(b, n, d)
    Y = block(X)

    assert X.shape == Y.shape
    assert sum([p.numel() for p in block.parameters()]) == sum(
        [
            d * 4 * (1 + d),  # input projection
            d * d + d,  # output projection
        ]
    )


def test_ONNX_export():
    b, n, d = 3, 100, 16
    block = ShepardsGatedAttention(d=d)

    X = torch.randn(b, n, d)
    mask = torch.rand(n, n) < 0.5
    Y = block(X, mask)

    f = BytesIO()
    torch.onnx.export(
        block,
        (X, mask),
        f,
        input_names=["X", "mask"],
        output_names=["Y"],
        dynamic_axes={"X": {0: "batch", 1: "token"}},
    )

    ort_session = ort.InferenceSession(f.getvalue())
    (out,) = ort_session.run(None, {"X": X.numpy(), "mask": mask.numpy()})
    Y2 = torch.from_numpy(out)
    torch.testing.assert_close(Y2, Y)
