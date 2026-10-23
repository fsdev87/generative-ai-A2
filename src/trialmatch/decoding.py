"""Greedy decoding three ways, all on top of QwenDecoder:

  greedy_uncached   reference: every step re-runs the full sequence from scratch
                    (fresh cache each step, nothing reused), O(n^2) work overall
  greedy_cached     prefill the prompt once, then feed one token per step and
                    reuse/extend the KV cache, O(n) work overall
  speculative       a small draft model proposes k tokens, the target verifies
                    them in one forward pass; output is identical to greedy_cached
                    on the target model

Each returns a DecodeResult with the generated token IDs, timings and memory.
Batch size is 1 throughout (no padding), which is the interactive screening case.
"""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class DecodeResult:
    tokens: list
    prefill_s: float = 0.0
    decode_s: float = 0.0
    peak_memory_mb: float | None = None
    cache_mb: float | None = None
    stats: dict = field(default_factory=dict)

    @property
    def total_s(self):
        return self.prefill_s + self.decode_s

    @property
    def tokens_per_s(self):
        return len(self.tokens) / self.total_s if self.total_s else 0.0

    @property
    def decode_tokens_per_s(self):
        # The first token comes out of prefill; the rest out of decode steps.
        return (len(self.tokens) - 1) / self.decode_s if self.decode_s and len(self.tokens) > 1 else 0.0

    def as_dict(self):
        return {"n_tokens": len(self.tokens), "prefill_s": round(self.prefill_s, 5),
                "decode_s": round(self.decode_s, 5), "total_s": round(self.total_s, 5),
                "tokens_per_s": round(self.tokens_per_s, 2),
                "decode_tokens_per_s": round(self.decode_tokens_per_s, 2),
                "peak_memory_mb": self.peak_memory_mb, "cache_mb": self.cache_mb, **self.stats}


class _Timer:
    """Wall-clock timing that waits for queued GPU work before reading the clock."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"

    def now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()


def _start_memory(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device):
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return None


def _as_ids(prompt_ids, device):
    ids = torch.as_tensor(prompt_ids, device=device)
    return ids[None] if ids.dim() == 1 else ids


def greedy_uncached(dec, prompt_ids, max_new_tokens, eos_ids=()):
    ids = _as_ids(prompt_ids, dec.device)
    timer, out = _Timer(dec.device), []
    _start_memory(dec.device)

    t0 = timer.now()
    prefill_s = None
    for _ in range(max_new_tokens):
        cache = dec.new_cache(1, ids.shape[1])  # fresh: no reuse across steps
        logits = dec.forward(ids, cache, 0)
        token = int(logits[0, -1].argmax())
        if prefill_s is None:
            prefill_s = timer.now() - t0
        out.append(token)
        if token in eos_ids:
            break
        ids = torch.cat([ids, torch.tensor([[token]], device=dec.device)], dim=1)
    total = timer.now() - t0
    return DecodeResult(out, prefill_s, total - prefill_s, _peak_memory_mb(dec.device))


def greedy_cached(dec, prompt_ids, max_new_tokens, eos_ids=()):
    ids = _as_ids(prompt_ids, dec.device)
    timer, out = _Timer(dec.device), []
    _start_memory(dec.device)
    cache = dec.new_cache(1, ids.shape[1] + max_new_tokens)

    t0 = timer.now()
    logits = dec.forward(ids, cache, 0)               # prefill: whole prompt, fills cache
    token = int(logits[0, -1].argmax())
    out.append(token)
    t1 = timer.now()

    while len(out) < max_new_tokens and token not in eos_ids:
        step = torch.tensor([[token]], device=dec.device)
        logits = dec.forward(step, cache, cache.length)  # one token at position cache.length
        token = int(logits[0, -1].argmax())
        out.append(token)
    t2 = timer.now()
    return DecodeResult(out, t1 - t0, t2 - t1, _peak_memory_mb(dec.device),
                        round(cache.used_bytes() / 2**20, 2))


def speculative(target, draft, prompt_ids, max_new_tokens, k=4, eos_ids=()):
    """Greedy speculative decoding (Leviathan et al. 2023; Chen et al. 2023), greedy variant.

    Invariant between iterations: `seq` is prompt + committed output, and both
    caches hold valid K/V for every position of `seq` except the last token, which
    is the pending input of the next forward pass.
    """
    ids = _as_ids(prompt_ids, target.device)
    timer = _Timer(target.device)
    _start_memory(target.device)
    capacity = ids.shape[1] + max_new_tokens + k + 1
    t_cache, d_cache = target.new_cache(1, capacity), draft.new_cache(1, capacity)
    stats = {"iterations": 0, "drafted": 0, "accepted": 0, "target_forwards": 1,
             "draft_forwards": 0, "accepted_per_iteration": []}

    t0 = timer.now()
    # Prefill the target; its argmax is the first committed output token.
    logits = target.forward(ids, t_cache, 0)
    seq = ids[0].tolist() + [int(logits[0, -1].argmax())]
    prompt_len = ids.shape[1]
    prefill_s = timer.now() - t0

    def output():
        return seq[prompt_len:]

    while len(output()) < max_new_tokens and seq[-1] not in eos_ids:
        stats["iterations"] += 1
        # 1. Draft k tokens greedily. The draft first catches up on any committed
        #    tokens it has not processed yet (cache_len .. len(seq) - 1).
        n_draft = min(k, max_new_tokens - len(output()))
        pending = seq[d_cache.length:]
        drafts = []
        for _ in range(n_draft):
            step = torch.tensor([pending], device=target.device)
            d_logits = draft.forward(step, d_cache, d_cache.length)
            stats["draft_forwards"] += 1
            pending = [int(d_logits[0, -1].argmax())]
            drafts.append(pending[0])

        # 2. Verify: one target pass over [last committed token, d_1 .. d_k].
        #    Row i of the logits is the target's greedy choice after d_1 .. d_i.
        verify = torch.tensor([[seq[-1]] + drafts], device=target.device)
        v_logits = target.forward(verify, t_cache, len(seq) - 1)
        stats["target_forwards"] += 1
        target_choice = v_logits[0].argmax(dim=-1).tolist()  # length n_draft + 1

        # 3. Accept the longest prefix where draft == target, then append the
        #    target's own token at the first mismatch (or a bonus token if all match).
        n_ok = 0
        while n_ok < n_draft and drafts[n_ok] == target_choice[n_ok]:
            n_ok += 1
        new_tokens = drafts[:n_ok] + [target_choice[n_ok]]
        stats["drafted"] += n_draft
        stats["accepted"] += n_ok
        stats["accepted_per_iteration"].append(n_ok)

        # Stop at EOS or the token budget inside the accepted block.
        for j, tok in enumerate(new_tokens):
            if tok in eos_ids or len(output()) + j + 1 >= max_new_tokens:
                new_tokens = new_tokens[: j + 1]
                break
        seq.extend(new_tokens)

        # 4. Roll back cache entries of rejected drafts: keep everything but the
        #    new pending token. The draft may lag behind; it catches up next round.
        t_cache.crop(len(seq) - 1)
        d_cache.crop(len(seq) - 1)

    total = timer.now() - t0
    stats["acceptance_rate"] = round(stats["accepted"] / stats["drafted"], 4) if stats["drafted"] else 0.0
    stats["mean_accepted_per_iteration"] = (
        round(stats["accepted"] / stats["iterations"], 3) if stats["iterations"] else 0.0)
    stats["k"] = k
    return DecodeResult(output(), prefill_s, total - prefill_s, _peak_memory_mb(target.device),
                        stats=stats)
