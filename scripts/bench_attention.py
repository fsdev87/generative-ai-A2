"""Task 4 benchmark: numerical agreement and runtime/memory of attention implementations.

  reference   our reference_attention (materializes the full score matrix)
  tiled       our tiled_attention with online softmax (block sizes 64 and 128)
  torch_sdpa  torch.nn.functional.scaled_dot_product_attention (fused backend)

Correctness: max/mean |error| against a float64 reference over several shapes,
causal/non-causal and decode-style (Lq < Lk) cases, in float32 (and float16 on GPU).
Performance: median of repeated, synchronized runs after warm-up, at the attention
shape of Qwen2.5-1.5B (12 query heads, head_dim 128); peak memory on GPU.

Usage:
  python scripts/bench_attention.py --out results/benchmarks/attention_cpu.json [--quick]
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from trialmatch.attention import reference_attention, tiled_attention

CORRECTNESS_SHAPES = [  # (B, H, Lq, Lk, D)
    (1, 1, 16, 16, 64),
    (2, 4, 100, 100, 64),
    (1, 12, 257, 257, 128),
    (1, 12, 1, 300, 128),   # single decode step against a cache
    (2, 12, 7, 512, 128),   # speculative verification block
]


def torch_sdpa(q, k, v, causal):
    if causal and q.shape[-2] != k.shape[-2]:
        keep = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep.tril(k.shape[-2] - q.shape[-2]))
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


IMPLS = {
    "reference": lambda q, k, v, c: reference_attention(q, k, v, causal=c),
    "tiled_64": lambda q, k, v, c: tiled_attention(q, k, v, causal=c, block_q=64, block_k=64),
    "tiled_128": lambda q, k, v, c: tiled_attention(q, k, v, causal=c, block_q=128, block_k=128),
    "torch_sdpa": torch_sdpa,
}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def correctness(device, dtypes):
    rows = []
    for dtype in dtypes:
        for shape in CORRECTNESS_SHAPES:
            B, H, Lq, Lk, D = shape
            g = torch.Generator().manual_seed(0)
            q, k, v = (torch.randn(B, H, L, D, generator=g, dtype=torch.float64) for L in (Lq, Lk, Lk))
            for causal in (False, True):
                exact = reference_attention(q, k, v, causal=causal)
                for name, fn in IMPLS.items():
                    out = fn(q.to(device, dtype), k.to(device, dtype), v.to(device, dtype), causal)
                    err = (out.double().cpu() - exact).abs()
                    rows.append({"impl": name, "dtype": str(dtype).replace("torch.", ""),
                                 "shape": list(shape), "causal": causal,
                                 "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())})
    return rows


def timed(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times), times


def performance(device, dtype, seq_lens, warmup, repeats):
    rows = []
    B, H, D = 1, 12, 128
    for L in seq_lens:
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(B, H, L, D, generator=g).to(device, dtype) for _ in range(3))
        for name, fn in IMPLS.items():
            if device.type == "cuda":
                torch.cuda.empty_cache()
                sync(device)
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            try:
                median, times = timed(lambda: fn(q, k, v, True), device, warmup, repeats)
            except torch.OutOfMemoryError:
                rows.append({"impl": name, "seq_len": L, "error": "out of memory"})
                continue
            row = {"impl": name, "seq_len": L, "median_ms": round(1000 * median, 3),
                   "all_ms": [round(1000 * t, 3) for t in times],
                   # Size of a full fp32 score matrix, what tiling avoids materializing.
                   "score_matrix_mb": round(B * H * L * L * 4 / 2**20, 1)}
            if device.type == "cuda":
                row["peak_extra_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
            rows.append(row)
            print(row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtypes = [torch.float32] + ([torch.float16] if device.type == "cuda" else [])
    perf_dtype = torch.float16 if device.type == "cuda" else torch.float32
    seq_lens = [128, 512, 1024] if args.quick else [128, 256, 512, 1024, 2048, 4096]
    warmup, repeats = (1, 3) if args.quick else (3, 10)

    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "correctness_reference": "float64 reference_attention",
        "correctness": correctness(device, dtypes),
        "performance_dtype": str(perf_dtype),
        "performance_shape": "B=1, H=12, D=128, causal",
        "warmup": warmup, "repeats": repeats,
        "performance": performance(device, perf_dtype, seq_lens, warmup, repeats),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    worst = {}
    for r in result["correctness"]:
        key = (r["impl"], r["dtype"])
        worst[key] = max(worst.get(key, 0.0), r["max_abs_err"])
    for (impl, dtype), err in sorted(worst.items()):
        print(f"max |err| {impl:10s} {dtype:8s} {err:.2e}")


if __name__ == "__main__":
    main()
