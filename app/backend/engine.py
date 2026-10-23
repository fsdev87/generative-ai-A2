"""Model loading and inference for the app, built on the project's own decoders."""

import os
import threading
import time

import torch

from trialmatch.attention import reference_attention, tiled_attention
from trialmatch.decoding import PrefixCache, greedy_cached, greedy_uncached, speculative
from trialmatch.modeling import DRAFT_MODEL, TARGET_MODEL, load_model, load_tokenizer
from trialmatch.prompting import build_messages, parse_output

from .report import explain

MODES = {
    "cached": "KV cache + prefix reuse (default)",
    "cached_tiled": "KV cache with our tiled online-softmax attention",
    "uncached": "No cache: full recomputation every token (reference)",
    "speculative": "Speculative decoding with the 0.5B draft model",
}


class Settings:
    def __init__(self):
        self.target = os.getenv("TARGET_MODEL", TARGET_MODEL)
        self.adapter = os.getenv("ADAPTER") or None
        self.draft = os.getenv("DRAFT_MODEL", DRAFT_MODEL)
        self.draft_adapter = os.getenv("DRAFT_ADAPTER") or None
        self.dtype = getattr(torch, os.getenv("DTYPE", "float32"))
        self.max_prompt_tokens = int(os.getenv("MAX_PROMPT_TOKENS", "2048"))
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "64"))
        self.spec_k = int(os.getenv("SPEC_K", "4"))
        self.answer_order = os.getenv("ANSWER_ORDER", "evidence_first")  # must match the adapter
        self.threads = int(os.getenv("TORCH_THREADS", "0"))


class InputTooLong(ValueError):
    pass


class Engine:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.state, self.error, self.loaded_at = "loading", None, None
        self.lock = threading.Lock()  # one model, one request at a time
        self.draft_dec = None

    # --- lifecycle -----------------------------------------------------------

    def load(self):
        s = self.settings
        try:
            if s.threads:
                torch.set_num_threads(s.threads)
            t0 = time.perf_counter()
            self.tok = load_tokenizer(s.target)
            self.target_hf = load_model(s.target, adapter=s.adapter, dtype=s.dtype)
            from trialmatch.qwen_decoder import QwenDecoder

            self.decoders = {"sdpa": QwenDecoder(self.target_hf),
                             "tiled": QwenDecoder(self.target_hf, attention="tiled")}
            self.prefix = PrefixCache(self.decoders["sdpa"], s.max_prompt_tokens + s.max_new_tokens)
            self.eos_ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
            self.load_seconds = round(time.perf_counter() - t0, 1)
            self.state, self.loaded_at = "ready", time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:  # surfaced through /api/health instead of crashing the server
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def _draft(self):
        if self.draft_dec is None:
            from trialmatch.qwen_decoder import QwenDecoder

            draft_hf = load_model(self.settings.draft, adapter=self.settings.draft_adapter,
                                  dtype=self.settings.dtype)
            self.draft_dec = QwenDecoder(draft_hf)
        return self.draft_dec

    def status(self):
        s = self.settings
        info = {"state": self.state, "error": self.error, "target_model": s.target,
                "adapter": s.adapter, "draft_model": s.draft, "dtype": str(s.dtype).replace("torch.", ""),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "max_prompt_tokens": s.max_prompt_tokens, "max_new_tokens": s.max_new_tokens,
                "spec_k": s.spec_k, "modes": MODES, "draft_loaded": self.draft_dec is not None}
        if self.state == "ready":
            info["loaded_at"], info["load_seconds"] = self.loaded_at, self.load_seconds
        return info

    def provenance(self, mode):
        s = self.settings
        return {"target_model": s.target, "adapter": s.adapter or "none (base model)",
                "draft_model": s.draft if mode == "speculative" else None,
                "decoding": f"greedy, {MODES[mode]}", "max_new_tokens": s.max_new_tokens,
                "dtype": str(s.dtype).replace("torch.", ""),
                "device": "cuda" if torch.cuda.is_available() else "cpu"}

    # --- inference -----------------------------------------------------------

    def prompt_ids(self, sentences, ctype, ctext):
        record = {"sentences": sentences, "criterion_type": ctype, "criterion_text": ctext}
        text = self.tok.apply_chat_template(build_messages(record, self.settings.answer_order),
                                            add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.settings.max_prompt_tokens:
            # Never truncate: the cut could remove the evidence the decision depends on.
            raise InputTooLong(f"prompt is {len(ids)} tokens; the limit is "
                               f"{self.settings.max_prompt_tokens}. Shorten the note or criterion.")
        return ids

    def _decode(self, ids, mode, max_new_tokens, eos_ids, reuse_prefix=True):
        ids_t = torch.tensor([ids], device=self.decoders["sdpa"].device)
        if mode == "cached":
            return greedy_cached(self.decoders["sdpa"], ids_t, max_new_tokens, eos_ids,
                                 prefix=self.prefix if reuse_prefix else None)
        if mode == "cached_tiled":
            return greedy_cached(self.decoders["tiled"], ids_t, max_new_tokens, eos_ids)
        if mode == "uncached":
            return greedy_uncached(self.decoders["sdpa"], ids_t, max_new_tokens, eos_ids)
        if mode == "speculative":
            return speculative(self.decoders["sdpa"], self._draft(), ids_t, max_new_tokens,
                               k=self.settings.spec_k, eos_ids=eos_ids)
        raise ValueError(f"unknown mode {mode!r}")

    def screen_one(self, sentences, ctype, ctext, mode="cached"):
        ids = self.prompt_ids(sentences, ctype, ctext)
        with self.lock:
            result = self._decode(ids, mode, self.settings.max_new_tokens, self.eos_ids)
        raw = self.tok.decode(result.tokens, skip_special_tokens=True)
        label, evidence, parse_ok = parse_output(raw, ctype, [sid for sid, _ in sentences])
        by_id = dict((sid, text) for sid, text in sentences)
        evidence_items = [{"id": sid, "text": by_id[sid]} for sid in evidence]
        explanation, flags = explain(ctype, ctext, label, evidence_items, parse_ok)
        return {
            "criterion_type": ctype, "criterion_text": ctext,
            "label": label, "evidence": evidence_items, "explanation": explanation, "flags": flags,
            "raw_output": raw, "mode": mode,
            "timing": {k: v for k, v in result.as_dict().items() if k != "accepted_per_iteration"},
        }

    def benchmark(self, sentences, ctype, ctext, modes, max_new_tokens, stop_at_eos):
        ids = self.prompt_ids(sentences, ctype, ctext)
        eos = self.eos_ids if stop_at_eos else ()
        rows, reference = [], None
        with self.lock:
            for mode in modes:
                # Fresh cache for every mode so all of them pay for the full prefill.
                result = self._decode(ids, mode, max_new_tokens, eos, reuse_prefix=False)
                if reference is None:
                    reference = result.tokens
                row = {"mode": mode, "description": MODES[mode], **result.as_dict(),
                       "tokens_equal_first_mode": result.tokens == reference,
                       "text": self.tok.decode(result.tokens, skip_special_tokens=False)}
                row.pop("accepted_per_iteration", None)
                rows.append(row)
        return {"prompt_tokens": len(ids), "max_new_tokens": max_new_tokens, "rows": rows}

    def attention_check(self, seq_len, causal, block):
        """Live comparison of our tiled attention vs the reference and torch SDPA."""
        g = torch.Generator().manual_seed(seq_len)
        q, k, v = (torch.randn(1, 12, seq_len, 128, generator=g) for _ in range(3))
        exact = reference_attention(q.double(), k.double(), v.double(), causal=causal)
        rows = []
        impls = {
            "reference": lambda: reference_attention(q, k, v, causal=causal),
            "tiled": lambda: tiled_attention(q, k, v, causal=causal, block_q=block, block_k=block),
            "torch_sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal),
        }
        with self.lock:
            for name, fn in impls.items():
                fn()  # warm-up
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    out = fn()
                    times.append(time.perf_counter() - t0)
                err = (out.double() - exact).abs()
                rows.append({"impl": name, "median_ms": round(1000 * sorted(times)[1], 3),
                             "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())})
        return {"shape": [1, 12, seq_len, 128], "causal": causal, "block": block,
                "score_matrix_mb": round(12 * seq_len * seq_len * 4 / 2**20, 2), "rows": rows}
