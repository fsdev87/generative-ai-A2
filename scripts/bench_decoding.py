"""Tasks 3 and 5 benchmark: uncached vs KV-cached vs speculative greedy decoding.

Section "lengths" (fixed output lengths, EOS ignored so every mode generates the same
number of tokens): for several prompt lengths x output lengths, compare
  uncached      full recomputation every step (reference)
  cached        explicit KV cache
  hf_generate   transformers generate() with its own cache (external sanity check)
  spec_k{K}     speculative decoding with the draft model, K drafted tokens/round
Token equality of every mode against `cached` is recorded.

Section "task" (real screening prompts from the validation split, stop at EOS):
cached vs speculative on N prompts; equality, acceptance and speedup.

Timing: one warm-up run per configuration, then the median of --repeats runs, with
CUDA synchronization before every clock read. Peak memory via CUDA allocator stats.

Usage:
  python scripts/bench_decoding.py --out results/benchmarks/decoding_gpu.json \
      [--adapter path/to/lora] [--draft-adapter path] [--quick]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from trialmatch.data import read_jsonl
from trialmatch.decoding import greedy_cached, greedy_uncached, speculative
from trialmatch.modeling import DRAFT_MODEL, TARGET_MODEL, load_model, load_tokenizer, prompt_text
from trialmatch.prompting import ANSWER_ORDERS
from trialmatch.qwen_decoder import QwenDecoder

ROOT = Path(__file__).resolve().parents[1]


def median_run(fn, repeats):
    fn()  # warm-up
    runs = [fn() for _ in range(repeats)]
    best = sorted(runs, key=lambda r: r.total_s)[len(runs) // 2]
    summary = best.as_dict()
    summary["total_s_all"] = [round(r.total_s, 5) for r in runs]
    summary["total_s"] = round(statistics.median(r.total_s for r in runs), 5)
    return best.tokens, summary


def length_prompts(tok, val_records, targets, order):
    """Real screening prompts whose token counts are closest to each target length;
    targets beyond the dataset are built by concatenating several notes."""
    texts = [prompt_text(tok, r, order) for r in val_records]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    prompts = {}
    for target in targets:
        if target <= max(lens):
            i = min(range(len(lens)), key=lambda j: abs(lens[j] - target))
            ids = tok(texts[i], add_special_tokens=False)["input_ids"]
        else:
            ids = []
            for t in texts:
                ids += tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) >= target:
                    break
            ids = ids[:target]
        prompts[target] = torch.tensor([ids])
    return prompts


def hf_generate(model, ids, n):
    def run():
        from trialmatch.decoding import DecodeResult, _Timer
        timer = _Timer(model.device)
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = timer.now()
        # No EOS: generate exactly n tokens without stopping. (min_new_tokens would instead
        # suppress EOS logits and pick a different token whenever EOS is the argmax.)
        out = model.generate(ids.to(model.device), max_new_tokens=n, do_sample=False, eos_token_id=None,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
        total = timer.now() - t0
        peak = round(torch.cuda.max_memory_allocated() / 2**20, 1) if model.device.type == "cuda" else None
        return DecodeResult(out[0, ids.shape[1]:].tolist(), 0.0, total, peak)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default=TARGET_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter merged into the target")
    parser.add_argument("--draft", default=DRAFT_MODEL)
    parser.add_argument("--draft-adapter", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--task-prompts", type=int, default=40)
    parser.add_argument("--answer-order", choices=ANSWER_ORDERS, default="evidence_first")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    prompt_lens = [128, 512] if args.quick else [128, 384, 1024, 2048]
    output_lens = [16, 64] if args.quick else [16, 64, 128]
    ks = [2, 4] if args.quick else [2, 4, 6, 8]
    task_n = 6 if args.quick else args.task_prompts

    tok = load_tokenizer(args.target)
    target_hf = load_model(args.target, adapter=args.adapter)
    draft_hf = load_model(args.draft, adapter=args.draft_adapter)
    target, draft = QwenDecoder(target_hf), QwenDecoder(draft_hf)
    device = target.device
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dtype": str(target.dtype), "torch": torch.__version__,
        "target": args.target, "adapter": args.adapter,
        "draft": args.draft, "draft_adapter": args.draft_adapter, "answer_order": args.answer_order,
        "decoding": "greedy (argmax), batch size 1, repetition_penalty 1.0",
        "repeats": args.repeats, "warmup": 1,
        "weights_mb": {
            "target": round(sum(p.numel() * p.element_size() for p in target_hf.parameters()) / 2**20, 1),
            "draft": round(sum(p.numel() * p.element_size() for p in draft_hf.parameters()) / 2**20, 1),
        },
        "lengths": [], "task": {},
    }

    val = read_jsonl(ROOT / "data" / "processed" / "val.jsonl")
    prompts = length_prompts(tok, val, prompt_lens, args.answer_order)
    for p_len, ids in prompts.items():
        for n in output_lens:
            ids = ids.to(device)
            modes = {
                "cached": lambda: greedy_cached(target, ids, n),
                "uncached": lambda: greedy_uncached(target, ids, n),
                "hf_generate": hf_generate(target_hf, ids, n),
            }
            for k in ks:
                modes[f"spec_k{k}"] = lambda k=k: speculative(target, draft, ids, n, k=k)
            if args.quick or (p_len >= 2048 and n >= 128):
                modes.pop("uncached")  # quadratic; skipped where it would dominate runtime

            tokens, rows = {}, {}
            for name, fn in modes.items():
                tokens[name], rows[name] = median_run(fn, args.repeats)
            for name in rows:
                rows[name]["tokens_equal_cached"] = tokens[name] == tokens["cached"]
                if not rows[name]["tokens_equal_cached"]:
                    first = next(i for i, (a, b) in enumerate(zip(tokens[name], tokens["cached"])) if a != b)
                    rows[name]["first_mismatch_index"] = first
            entry = {"prompt_tokens": int(ids.shape[1]), "new_tokens": n, "modes": rows}
            result["lengths"].append(entry)
            print(json.dumps({"prompt": p_len, "new": n,
                              **{m: (r["total_s"], r["tokens_equal_cached"]) for m, r in rows.items()}}),
                  flush=True)

    # Real task: short JSON answers, stop at EOS.
    k_task = 4
    cached_rows, spec_rows, equal = [], [], 0
    for record in val[:task_n]:
        ids = tok(prompt_text(tok, record, args.answer_order), return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(device)
        a_tokens, a = median_run(lambda: greedy_cached(target, ids, 64, eos_ids), args.repeats)
        b_tokens, b = median_run(lambda: speculative(target, draft, ids, 64, k=k_task, eos_ids=eos_ids),
                                 args.repeats)
        equal += a_tokens == b_tokens
        cached_rows.append(a)
        spec_rows.append(b)
    drafted = sum(r["drafted"] for r in spec_rows)
    result["task"] = {
        "n_prompts": task_n, "k": k_task, "tokens_equal": equal,
        "cached_total_s": round(sum(r["total_s"] for r in cached_rows), 4),
        "spec_total_s": round(sum(r["total_s"] for r in spec_rows), 4),
        "acceptance_rate": round(sum(r["accepted"] for r in spec_rows) / drafted, 4) if drafted else 0.0,
        "mean_new_tokens": round(statistics.mean(r["n_tokens"] for r in cached_rows), 2),
    }
    result["task"]["speedup"] = round(result["task"]["cached_total_s"] / result["task"]["spec_total_s"], 3)
    print("task:", result["task"], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
