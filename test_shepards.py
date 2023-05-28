from shepards import autoregressive_mask, shepards_MHA
import torch


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

    K = torch.randn(b, h, n, d_qk)
    V = torch.randn(b, h, n, d_v)

    order = torch.randperm(n)
    Q, T = K[:, :, order], V[:, :, order]

    # Test that attention works.
    Y = shepards_MHA(Q, K, V)
    T = V[:, :, order]
    torch.testing.assert_close(Y, T)

    # Test that instances and heads are independent.
    for batch in range(b):
        for head in range(h):
            Y2 = shepards_MHA(Q[batch, head], K[batch, head], V[batch, head])
            torch.testing.assert_close(Y2, Y[batch, head])
