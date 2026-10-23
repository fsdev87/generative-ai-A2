"""A hand-written Qwen2 forward pass with an explicit, preallocated KV cache.

The Hugging Face model is used only as a container of trained weights (embedding,
linear projections, norm weights). Everything that matters for cached decoding is
implemented here:

  * KVCache: per-layer K/V buffers of shape (B, H_kv, max_len, D) written in place;
    `length` marks how many positions are valid, so growth is O(1) per token (no
    torch.cat reallocation) and rollback for speculative decoding is `crop(n)`.
  * Rotary position embeddings applied at explicit absolute positions
    start_pos .. start_pos + T - 1, so new queries/keys rotate exactly as they
    would inside a full recomputation.
  * Causal masking aligned to the bottom-right of the (T, start_pos + T) score
    matrix: new token i may see cached positions and new tokens <= i.
  * Grouped-query attention: K/V are cached with H_kv heads and expanded to the
    query heads only at attention time, so the cache is H / H_kv times smaller.

Architecture (Qwen2): x -> [RMSNorm -> attention(q,k,v with bias; o without) -> +residual
-> RMSNorm -> SwiGLU MLP -> +residual] x N -> RMSNorm -> lm_head.
"""

import torch
import torch.nn.functional as F

from .attention import reference_attention, repeat_kv, tiled_attention


class KVCache:
    def __init__(self, n_layers, batch, n_kv_heads, head_dim, max_len, dtype, device):
        shape = (batch, n_kv_heads, max_len, head_dim)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_len = max_len
        self.length = 0

    def write(self, layer, k, v, start):
        end = start + k.shape[2]
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: need {end} positions, capacity {self.max_len}")
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def crop(self, length):
        """Discard positions >= length (stale entries are simply overwritten later)."""
        self.length = min(self.length, length)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def used_bytes(self):
        return self.nbytes() * self.length // self.max_len


class QwenDecoder:
    """Explicit forward pass over the weights of a Hugging Face Qwen2ForCausalLM."""

    def __init__(self, hf_model, attention="sdpa"):
        cfg = hf_model.config
        self.model = hf_model
        self.layers = hf_model.model.layers
        self.embed = hf_model.model.embed_tokens
        self.final_norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        rope = getattr(cfg, "rope_parameters", None) or {}
        self.rope_theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype
        self.attention = attention
        self.inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim
        ))

    def new_cache(self, batch, max_len):
        return KVCache(len(self.layers), batch, self.n_kv_heads, self.head_dim,
                       max_len, self.dtype, self.device)

    # --- building blocks -------------------------------------------------------

    def _rms_norm(self, x, weight):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return weight * x32.to(x.dtype)

    def _rope(self, positions):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]   # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)                       # (T, D)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _apply_rope(x, cos, sin):
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin

    def _attend(self, q, k, v, causal):
        if self.attention == "tiled":
            return tiled_attention(q, k, v, causal=causal)
        if self.attention == "reference":
            return reference_attention(q, k, v, causal=causal)
        # PyTorch SDPA (fused when available). Its is_causal is top-left aligned, so
        # with cached keys we pass the bottom-right mask explicitly.
        if not causal:
            return F.scaled_dot_product_attention(q, k, v)
        lq, lk = q.shape[-2], k.shape[-2]
        if lq == lk:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        keep = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=lk - lq)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

    # --- forward ---------------------------------------------------------------

    @torch.no_grad()
    def forward(self, input_ids, cache, start_pos):
        """Run tokens at absolute positions start_pos.. through the model.

        Writes their K/V into `cache` at the same positions and returns logits of
        shape (B, T, vocab). Cache entries at positions < start_pos must be valid.
        """
        B, T = input_ids.shape
        if start_pos > cache.length:
            raise ValueError(f"start_pos {start_pos} beyond valid cache length {cache.length}")
        positions = torch.arange(start_pos, start_pos + T, device=self.device)
        cos, sin = self._rope(positions)
        causal = T > 1  # a single new token may attend to every cached position

        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            attn, mlp = layer.self_attn, layer.mlp
            x = self._rms_norm(h, layer.input_layernorm.weight)

            q = attn.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

            k_all, v_all = cache.write(i, k, v, start_pos)
            n_rep = self.n_heads // self.n_kv_heads
            o = self._attend(q, repeat_kv(k_all, n_rep), repeat_kv(v_all, n_rep), causal)
            h = h + attn.o_proj(o.transpose(1, 2).reshape(B, T, -1))

            x = self._rms_norm(h, layer.post_attention_layernorm.weight)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

        cache.length = start_pos + T
        return self.lm_head(self._rms_norm(h, self.final_norm.weight))
