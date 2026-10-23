"""Reference and tiled (online-softmax) scaled dot-product attention.

Shapes follow PyTorch SDPA: q (B, H, Lq, D), k and v (B, H, Lk, D) -> (B, H, Lq, D).

Causal masking is "bottom-right aligned": query i (0-based) may attend key j iff
j <= i + (Lk - Lq). With Lq == Lk this is the usual lower triangle; with Lq < Lk
it covers decoding with a KV cache, where the new queries sit at the end of the
key sequence.

An optional `mask` broadcastable to (B, H, Lq, Lk) is either boolean (True = may
attend) or additive float (0 / -inf), matching what Hugging Face passes to
attention backends.
"""

import math

import torch


def _causal_keep(lq, lk, device, q_start=0, k_start=0, q_len=None, k_len=None):
    """Boolean (q_len, k_len) block of the bottom-right-aligned causal mask."""
    q_len = lq if q_len is None else q_len
    k_len = lk if k_len is None else k_len
    q_idx = torch.arange(q_start, q_start + q_len, device=device)[:, None] + (lk - lq)
    k_idx = torch.arange(k_start, k_start + k_len, device=device)[None, :]
    return k_idx <= q_idx


def _apply_mask(scores, mask):
    if mask is None:
        return scores
    if mask.dtype == torch.bool:
        return scores.masked_fill(~mask, float("-inf"))
    return scores + mask


def reference_attention(q, k, v, causal=False, mask=None, scale=None):
    """Materializes the full (Lq, Lk) score matrix; the correctness oracle."""
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

    scores = (q.to(compute_dtype) @ k.to(compute_dtype).transpose(-2, -1)) * scale
    if causal:
        keep = _causal_keep(q.shape[-2], k.shape[-2], q.device)
        scores = scores.masked_fill(~keep, float("-inf"))
    scores = _apply_mask(scores, mask)

    # Numerically stable softmax: subtracting the row max keeps exp() <= 1.
    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
    weights = torch.exp(scores - row_max)
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / torch.where(denom == 0, torch.ones_like(denom), denom)
    return (weights @ v.to(compute_dtype)).to(q.dtype)


def tiled_attention(q, k, v, causal=False, mask=None, scale=None, block_q=64, block_k=64):
    """Blocked attention with an online softmax (FlashAttention-style forward pass).

    For each query block we stream over key/value blocks and keep, per query row:
      m   running maximum of the scores seen so far
      l   running softmax denominator, expressed relative to m
      acc running numerator (weights @ V), expressed relative to m
    When a new block raises the maximum from m to m_new, previous l and acc are
    rescaled by exp(m - m_new). The full (Lq, Lk) score matrix is never stored:
    peak extra memory is O(block_q * block_k) per (batch, head).
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    B, H, Lq, D = q.shape
    Lk = k.shape[-2]
    out = torch.empty(B, H, Lq, v.shape[-1], dtype=q.dtype, device=q.device)

    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        q_blk = q[:, :, qs:qe].to(compute_dtype) * scale
        rows = qe - qs
        m = torch.full((B, H, rows, 1), float("-inf"), dtype=compute_dtype, device=q.device)
        l = torch.zeros((B, H, rows, 1), dtype=compute_dtype, device=q.device)
        acc = torch.zeros((B, H, rows, v.shape[-1]), dtype=compute_dtype, device=q.device)

        # Under causal masking, keys beyond the last query's diagonal are never visible.
        k_limit = min(Lk, (qe - 1) + (Lk - Lq) + 1) if causal else Lk
        for ks in range(0, k_limit, block_k):
            ke = min(ks + block_k, k_limit)
            s = q_blk @ k[:, :, ks:ke].to(compute_dtype).transpose(-2, -1)
            if causal:
                keep = _causal_keep(Lq, Lk, q.device, qs, ks, rows, ke - ks)
                s = s.masked_fill(~keep, float("-inf"))
            if mask is not None:
                s = _apply_mask(s, mask[..., qs:qe, ks:ke] if mask.shape[-2] > 1 else mask[..., ks:ke])

            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # Rows that have seen only masked keys keep m_new = -inf; use 0 to avoid inf - inf.
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            correction = torch.exp(m - m_safe)
            l = l * correction + p.sum(dim=-1, keepdim=True)
            acc = acc * correction + p @ v[:, :, ks:ke].to(compute_dtype)
            m = m_new

        out[:, :, qs:qe] = (acc / torch.where(l == 0, torch.ones_like(l), l)).to(q.dtype)
    return out


def repeat_kv(x, n_rep):
    """Expand grouped-query K/V heads (B, Hkv, L, D) to (B, Hkv * n_rep, L, D)."""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


def make_hf_attention(kernel):
    """Wrap `reference_attention` or `tiled_attention` as a Hugging Face attention backend."""

    def hf_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        n_rep = query.shape[1] // key.shape[1]
        key, value = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
        if attention_mask is not None:
            out = kernel(query, key, value, causal=False,
                         mask=attention_mask[..., : key.shape[-2]], scale=scaling)
        else:
            out = kernel(query, key, value, causal=query.shape[-2] > 1, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    return hf_attention


def register_hf_backends():
    """Make attn_implementation="tiled" / "reference_sdpa" available to transformers models.

    transformers builds the attention mask per backend name and passes None to an
    unknown backend, which would silently ignore padding; registering the SDPA mask
    builder makes it pass a boolean (B, 1, Lq, Lk) keep-mask instead.
    """
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    for name, kernel in [("tiled", tiled_attention), ("reference_sdpa", reference_attention)]:
        AttentionInterface.register(name, make_hf_attention(kernel))
        AttentionMaskInterface.register(name, sdpa_mask)
