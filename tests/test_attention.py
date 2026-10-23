import pytest
import torch
import torch.nn.functional as F

from trialmatch.attention import reference_attention, repeat_kv, tiled_attention

SHAPES = [  # (B, H, Lq, Lk, D)
    (1, 1, 1, 1, 8),
    (1, 2, 7, 7, 16),
    (2, 4, 64, 64, 32),
    (2, 3, 100, 100, 64),  # not a multiple of the block size
    (1, 2, 5, 37, 16),     # decode-style: few queries over a longer cached key sequence
    (1, 2, 1, 130, 64),    # single-token decode step
]


def rand(B, H, Lq, Lk, D, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, Lq, D, generator=g, dtype=dtype)
    k = torch.randn(B, H, Lk, D, generator=g, dtype=dtype)
    v = torch.randn(B, H, Lk, D, generator=g, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_reference_matches_torch_sdpa(shape, causal):
    B, H, Lq, Lk, D = shape
    q, k, v = rand(*shape)
    ours = reference_attention(q, k, v, causal=causal)
    if causal and Lq != Lk:
        # PyTorch's is_causal is top-left aligned; build our bottom-right mask explicitly.
        keep = torch.ones(Lq, Lk, dtype=torch.bool).tril(diagonal=Lk - Lq)
        theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    else:
        theirs = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(ours, theirs, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("blocks", [(1, 1), (4, 8), (16, 16), (64, 64)])
def test_tiled_matches_reference(shape, causal, blocks):
    q, k, v = rand(*shape)
    ref = reference_attention(q, k, v, causal=causal)
    tiled = tiled_attention(q, k, v, causal=causal, block_q=blocks[0], block_k=blocks[1])
    torch.testing.assert_close(tiled, ref, atol=1e-10, rtol=1e-10)


def test_tiled_float32_tolerance():
    q, k, v = rand(2, 4, 96, 96, 64, dtype=torch.float32)
    torch.testing.assert_close(
        tiled_attention(q, k, v, causal=True, block_q=32, block_k=32),
        reference_attention(q, k, v, causal=True), atol=1e-5, rtol=1e-5,
    )


def test_online_softmax_is_stable_for_large_scores():
    q, k, v = rand(1, 1, 16, 16, 8)
    q = q * 1e3  # naive exp() would overflow
    out = tiled_attention(q, k, v, causal=True, block_q=4, block_k=4)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference_attention(q, k, v, causal=True))


def test_padding_mask():
    q, k, v = rand(2, 2, 10, 10, 16)
    keep = torch.ones(2, 1, 10, 10, dtype=torch.bool).tril()
    keep[0, :, :, :3] = False   # left padding on the first sequence...
    keep[0, :, :3, :3] = torch.eye(3, dtype=torch.bool)  # ...padded rows attend to themselves
    additive = torch.zeros(keep.shape, dtype=q.dtype).masked_fill(~keep, float("-inf"))
    ref = reference_attention(q, k, v, mask=keep)
    torch.testing.assert_close(tiled_attention(q, k, v, mask=keep, block_q=4, block_k=3), ref)
    torch.testing.assert_close(tiled_attention(q, k, v, mask=additive, block_q=4, block_k=3), ref)


def test_repeat_kv_matches_repeat_interleave():
    x = torch.randn(2, 3, 5, 4)
    torch.testing.assert_close(repeat_kv(x, 4), x.repeat_interleave(4, dim=1))
