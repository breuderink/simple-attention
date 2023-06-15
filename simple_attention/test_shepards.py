import torch
from .shepards import shepards_MHA


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

    Q = torch.randn(h, b, n, d_qk)
    K = torch.randn(h, b, n, d_qk)
    V = torch.randn(h, b, n, d_v)

    Y = shepards_MHA(Q, K, V)

    # Test that instances and heads are independent.
    for head in range(h):
        for batch in range(b):
            Y2 = shepards_MHA(Q[head, batch], K[head, batch], V[head, batch])
            torch.testing.assert_close(Y2, Y[head, batch])


def test_masked_attention():
    b, n_q, n_kv, d = 2, 3, 5, 16

    Q = torch.randn(b, n_q, d)
    K = torch.randn(b, n_kv, d)
    V = torch.randn(b, n_kv, d)

    mask = torch.rand(b, n_q, n_kv) < 0.5

    Y = shepards_MHA(Q, K, V, mask=mask)

    for i in range(n_q):
        # Modify a single token.
        V2 = V.clone()
        V2[:, i] = 0

        # Recompute attention.
        Y2 = shepards_MHA(Q, K, V2, mask=mask)

        # Test correspondence between mask and unchanged values.
        unaffected = torch.all(torch.isclose(Y2, Y), dim=2)
        assert (unaffected == mask[:, :, i]).all()
