"""Generate all report figures (PDF) from the JSON files in results/.

Usage: python scripts/make_figures.py   ->  report/figures/*.pdf
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = 3.45  # IEEE column width in inches
COLORS = {"main": "#2563eb", "label_first": "#059669", "oversample": "#c2410c", "epochs5": "#7c3aed",
          "draft_lora": "#64748b"}
NAMES = {"main": "evidence-first (main)", "label_first": "label-first (final)",
         "oversample": "oversampling", "epochs5": "5 epochs", "draft_lora": "0.5B draft"}


def load(path):
    return json.loads((RES / path).read_text())


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def smooth(y, k=9):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def training_curves():
    runs = {"main": "lora_r16", "label_first": "label_first", "oversample": "oversample", "epochs5": "epochs5"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL * 2, 2.1))
    for key, folder in runs.items():
        hist = load(f"training/{folder}/history.json")
        a.plot([h["epoch"] for h in hist], smooth([h["loss"] for h in hist]), color=COLORS[key],
               label=NAMES[key], lw=1)
        val = load(f"training/{folder}/val_by_epoch.json")
        b.plot([v["epoch"] for v in val], [v["macro_f1"] for v in val], "o-", color=COLORS[key],
               label=NAMES[key], lw=1, ms=3)
    a.set(xlabel="epoch", ylabel="train loss (answer tokens, smoothed)", yscale="log")
    b.set(xlabel="epoch", ylabel="validation macro-F1")
    b.axhline(load("eval/zeroshot_val/metrics.json")["overall"]["classification"]["macro_f1"],
              color="k", ls=":", lw=1, label="zero-shot")
    b.legend(frameon=False, ncol=1)
    save(fig, "training_curves")


def confusion(before, after, name):
    fig, axes = plt.subplots(2, 2, figsize=(COL * 2, 4.2))
    for row, (title, path) in enumerate([("zero-shot", before), ("fine-tuned", after)]):
        m = load(path)
        for col, subset in enumerate(["inclusion", "exclusion"]):
            cm = m[subset]["classification"]["confusion_matrix"]
            mat = np.array(cm["matrix"])
            ax = axes[row, col]
            ax.imshow(mat, cmap="Blues")
            short = [l.replace("not enough information", "NEI").replace("not applicable", "N/A")
                     for l in cm["labels"]]
            ax.set_xticks(range(len(short)), short, rotation=30, ha="right")
            ax.set_yticks(range(len(short)), short)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > mat.max() / 2 else "black")
            ax.set_title(f"{title} - {subset}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("gold")
    save(fig, name)


def attention():
    data = load("benchmarks/attention_gpu.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL * 2, 2.1))
    impls = sorted({r["impl"] for r in data["performance"]})
    for impl in impls:
        rows = [r for r in data["performance"] if r["impl"] == impl and "median_ms" in r]
        a.plot([r["seq_len"] for r in rows], [r["median_ms"] for r in rows], "o-", ms=3, lw=1, label=impl)
        if rows and "peak_extra_mb" in rows[0]:
            b.plot([r["seq_len"] for r in rows], [r["peak_extra_mb"] for r in rows], "o-", ms=3, lw=1, label=impl)
    for ax, ylabel in [(a, "median time (ms)"), (b, "peak extra memory (MB)")]:
        ax.set(xlabel="sequence length", ylabel=ylabel, xscale="log", yscale="log")
        ax.set_xticks([128, 256, 512, 1024, 2048, 4096], ["128", "256", "512", "1k", "2k", "4k"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.legend(frameon=False)
    save(fig, "attention_perf")


def kv_cache():
    data = load("benchmarks/decoding_gpu_draft_base.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL * 2, 2.1))
    for n_new, marker in [(16, "s"), (64, "o"), (128, "^")]:
        entries = [e for e in data["lengths"] if e["new_tokens"] == n_new]
        p = [e["prompt_tokens"] for e in entries]
        a.plot(p, [e["modes"]["cached"]["total_s"] for e in entries], marker + "-", color="#2563eb",
               ms=3, lw=1, label=f"cached, {n_new} new")
        unc = [(e["prompt_tokens"], e["modes"]["uncached"]["total_s"]) for e in entries if "uncached" in e["modes"]]
        if unc:
            a.plot(*zip(*unc), marker + "--", color="#c2410c", ms=3, lw=1, label=f"uncached, {n_new} new")
        b.plot(p, [e["modes"]["cached"]["peak_memory_mb"] for e in entries], marker + "-", color="#2563eb",
               ms=3, lw=1, label=f"cached, {n_new} new")
        if unc:
            b.plot([e["prompt_tokens"] for e in entries if "uncached" in e["modes"]],
                   [e["modes"]["uncached"]["peak_memory_mb"] for e in entries if "uncached" in e["modes"]],
                   marker + "--", color="#c2410c", ms=3, lw=1, label=f"uncached, {n_new} new")
    a.set(xlabel="prompt tokens", ylabel="end-to-end time (s)", xscale="log", yscale="log")
    b.set(xlabel="prompt tokens", ylabel="peak GPU memory (MB)", xscale="log")
    for ax in (a, b):
        ticks = [e["prompt_tokens"] for e in data["lengths"] if e["new_tokens"] == 16]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "kv_cache")


def speculative():
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL * 2, 2.1))
    for label, path, color in [("base 0.5B draft", "benchmarks/decoding_gpu_draft_base.json", "#64748b"),
                               ("LoRA 0.5B draft", "benchmarks/decoding_gpu_draft_lora.json", "#059669")]:
        data = load(path)
        entries = [e for e in data["lengths"] if e["new_tokens"] == max(x["new_tokens"] for x in data["lengths"])]
        ks = sorted(int(m[6:]) for m in entries[0]["modes"] if m.startswith("spec_k"))
        speed = [np.mean([e["modes"]["cached"]["total_s"] / e["modes"][f"spec_k{k}"]["total_s"] for e in entries])
                 for k in ks]
        acc = [np.mean([e["modes"][f"spec_k{k}"]["acceptance_rate"] for e in entries]) for k in ks]
        a.plot(ks, speed, "o-", color=color, ms=3, lw=1, label=label)
        b.plot(ks, acc, "o-", color=color, ms=3, lw=1, label=label)
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.set(xlabel="draft length k", ylabel="speed-up vs KV-cached greedy")
    b.set(xlabel="draft length k", ylabel="draft acceptance rate", ylim=(0, 1))
    a.legend(frameon=False)
    save(fig, "speculative")


if __name__ == "__main__":
    for fn, needs in [(training_curves, "training/label_first/history.json"),
                      (lambda: confusion("eval/zeroshot_test/metrics.json", "eval/final_test/metrics.json",
                                         "confusion_test"), "eval/final_test/metrics.json"),
                      (attention, "benchmarks/attention_gpu.json"),
                      (kv_cache, "benchmarks/decoding_gpu_draft_base.json"),
                      (speculative, "benchmarks/decoding_gpu_draft_lora.json")]:
        if (RES / needs).exists():
            fn()
        else:
            print("skipped (missing)", needs)
