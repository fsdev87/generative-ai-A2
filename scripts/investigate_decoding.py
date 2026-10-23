"""Investigate (1) cached-vs-uncached greedy mismatches and (2) the draft/target cost ratio.

1. For a prompt where the fp16 benchmark found cached and uncached greedy outputs diverging,
   decode step by step with both paths, recording at every step the top-2 logit margin and the
   max |logit difference| between the paths. At the first divergent step we report both paths'
   top-2 candidates. The same prompt is then decoded in fp32 to check whether the divergence
   disappears (i.e. it is a precision near-tie, not a cache bug).
2. Time single-token decode steps of the target and the draft with our decoder to obtain the cost
   ratio c used in the speculative-decoding speed-up model of Leviathan et al. (2023):
       E[speed-up] = (1 - a^(k+1)) / ((1 - a) * (c*k + 1))

Usage:
  python scripts/investigate_decoding.py --adapter A --draft-adapter D --answer-order label_first \
      --out results/benchmarks/decoding_investigation.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from trialmatch.data import read_jsonl
from trialmatch.modeling import DRAFT_MODEL, TARGET_MODEL, load_model, load_tokenizer, prompt_text
from trialmatch.prompting import ANSWER_ORDERS
from trialmatch.qwen_decoder import QwenDecoder

ROOT = Path(__file__).resolve().parents[1]


def trace(dec, ids, n_new):
    """Greedy decode n_new tokens with and without cache, recording per-step logits."""
    cache = dec.new_cache(1, ids.shape[1] + n_new)
    seq = ids
    steps = []
    logits_c = dec.forward(ids, cache, 0)[0, -1].float()
    for step in range(n_new):
        logits_u = dec.forward(seq, dec.new_cache(1, seq.shape[1]), 0)[0, -1].float()
        top_c, top_u = logits_c.topk(2), logits_u.topk(2)
        steps.append({
            "step": step,
            "cached_top2": [[int(i), float(v)] for v, i in zip(*top_c)],
            "uncached_top2": [[int(i), float(v)] for v, i in zip(*top_u)],
            "max_abs_logit_diff": float((logits_c - logits_u).abs().max()),
            "margin_cached": float(top_c.values[0] - top_c.values[1]),
        })
        if int(top_c.indices[0]) != int(top_u.indices[0]):
            return steps, step
        token = top_c.indices[:1].view(1, 1)
        seq = torch.cat([seq, token], dim=1)
        logits_c = dec.forward(token, cache, cache.length)[0, -1].float()
    return steps, None


@torch.no_grad()
def step_time(dec, prompt_len, repeats=50):
    ids = torch.randint(100, 1000, (1, prompt_len), device=dec.device)
    cache = dec.new_cache(1, prompt_len + repeats + 10)
    dec.forward(ids, cache, 0)
    token = ids[:, -1:]
    for _ in range(5):  # warm-up
        dec.forward(token, cache, cache.length)
    cache.crop(prompt_len)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec.forward(token, cache, cache.length)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--draft-adapter", default=None)
    parser.add_argument("--answer-order", choices=ANSWER_ORDERS, default="evidence_first")
    parser.add_argument("--n-new", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tok = load_tokenizer(TARGET_MODEL)
    val = read_jsonl(ROOT / "data" / "processed" / "val.jsonl")
    # The benchmark's shortest real prompt (closest to 128 tokens) is where fp16 diverged.
    texts = [prompt_text(tok, r, args.answer_order) for r in val]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    i = min(range(len(lens)), key=lambda j: abs(lens[j] - 128))
    ids = tok(texts[i], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

    result = {"prompt_tokens": int(ids.shape[1]), "n_new": args.n_new, "precision": {}}
    for dtype in (torch.float16, torch.float32):
        model = load_model(TARGET_MODEL, adapter=args.adapter, dtype=dtype)
        steps, diverged = trace(QwenDecoder(model), ids, args.n_new)
        entry = {"diverged_at_step": diverged, "steps_compared": len(steps),
                 "max_abs_logit_diff_before_divergence": max(
                     (s["max_abs_logit_diff"] for s in steps[:-1 if diverged is not None else None]), default=0.0),
                 "min_margin_before_divergence": min(
                     (s["margin_cached"] for s in steps[:-1 if diverged is not None else None]), default=None)}
        if diverged is not None:
            d = steps[diverged]
            entry["at_divergence"] = {**d, "cached_tokens": [tok.decode([t]) for t, _ in d["cached_top2"]],
                                      "uncached_tokens": [tok.decode([t]) for t, _ in d["uncached_top2"]]}
        result["precision"][str(dtype).replace("torch.", "")] = entry
        print(str(dtype), json.dumps(entry, indent=1)[:1500], flush=True)
        del model
        torch.cuda.empty_cache()

    target = QwenDecoder(load_model(TARGET_MODEL, adapter=args.adapter, dtype=torch.float16))
    draft = QwenDecoder(load_model(DRAFT_MODEL, adapter=args.draft_adapter, dtype=torch.float16))
    costs = {}
    for prompt_len in (300, 1000, 2000):
        t, d = step_time(target, prompt_len), step_time(draft, prompt_len)
        costs[prompt_len] = {"target_ms": round(1000 * t, 3), "draft_ms": round(1000 * d, 3), "c": round(d / t, 3)}
    result["decode_step_cost"] = costs
    c = statistics.mean(v["c"] for v in costs.values())
    result["predicted_speedup"] = {
        f"a={a},k={k}": round((1 - a ** (k + 1)) / ((1 - a) * (c * k + 1)), 3)
        for a in (0.75, 0.93) for k in (2, 4, 8)
    }
    result["c_mean"] = round(c, 3)
    print(json.dumps({k: result[k] for k in ("decode_step_cost", "c_mean", "predicted_speedup")}, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
