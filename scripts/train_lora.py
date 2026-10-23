"""LoRA fine-tuning of the target model on the prescribed training split.

Training objective: causal-LM cross-entropy on the assistant answer tokens only
({"evidence": [...], "label": ...}<|im_end|>); prompt tokens are masked with -100.
After every epoch the adapter is saved and evaluated on the validation split with
greedy decoding; the epoch with the best validation macro-F1 is kept as `best/`.

Outputs in --out:
  config.json          all hyperparameters and environment info
  history.json         per-step training loss / learning rate
  val_by_epoch.json    validation loss and generation metrics per epoch
  baseline_val/        (with --baseline) zero-shot metrics of the untuned model
  epoch_<k>/           LoRA adapter after epoch k
  best/                adapter of the best validation epoch + its predictions

Usage:
  python scripts/train_lora.py --out runs/lora_r16 --baseline
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from trialmatch.data import read_jsonl, write_jsonl
from trialmatch.evaluation import evaluate, headline
from trialmatch.modeling import TARGET_MODEL, load_model, load_tokenizer, pick_device, prompt_text
from trialmatch.prompting import build_target

ROOT = Path(__file__).resolve().parents[1]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, record, max_length):
    """Prompt + answer token IDs with the prompt masked out of the loss."""
    prompt_ids = tok(prompt_text(tok, record), add_special_tokens=False)["input_ids"]
    answer_ids = tok(build_target(record) + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        # Never truncate silently: truncation could remove patient evidence sentences.
        raise ValueError(
            f"annotation {record['annotation_id']} needs {len(prompt_ids) + len(answer_ids)} "
            f"tokens > max_length={max_length}"
        )
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def collate(examples, pad_id):
    """Right-pad a batch for teacher-forced training."""
    width = max(len(e["input_ids"]) for e in examples)
    input_ids, labels, attention = [], [], []
    for e in examples:
        pad = width - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [pad_id] * pad)
        labels.append(e["labels"] + [-100] * pad)
        attention.append([1] * len(e["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


@torch.no_grad()
def validation_loss(model, batches, device, amp_dtype):
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(**batch).loss
        n_tokens = int((batch["labels"] != -100).sum())
        total += float(loss) * n_tokens
        count += n_tokens
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Effective batch 8. Batch 4 runs out of memory on a 15 GB T4: the fp32 logits
    # alone are 4 x 651 x 151,936 x 4 B = 1.6 GB per copy.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--baseline", action="store_true", help="also evaluate the untuned model on val")
    parser.add_argument("--limit-train", type=int, default=None, help="smoke tests only")
    parser.add_argument("--limit-val", type=int, default=None, help="smoke tests only")
    parser.add_argument("--longest-first", action="store_true",
                        help="with --limit-train: take the longest examples (memory preflight)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    amp_dtype = torch.float16 if device.type == "cuda" else None

    train_records = read_jsonl(ROOT / "data" / "processed" / "train.jsonl")
    if args.longest_first:
        train_records.sort(key=lambda r: -sum(len(t) for _, t in r["sentences"]) - len(r["criterion_text"]))
    train_records = train_records[: args.limit_train]
    val_records = read_jsonl(ROOT / "data" / "processed" / "val.jsonl")[: args.limit_val]

    tok = load_tokenizer(args.model)
    model = load_model(args.model, device=device)

    config = {**vars(args), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "torch": torch.__version__, "lora_targets": LORA_TARGETS,
              "n_train": len(train_records), "n_val": len(val_records)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    if args.baseline:
        print("== zero-shot baseline on val", flush=True)
        t0 = time.perf_counter()
        preds, report = evaluate(model, tok, val_records, batch_size=args.eval_batch_size)
        report["seconds"] = round(time.perf_counter() - t0, 1)
        (out / "baseline_val").mkdir(exist_ok=True)
        write_jsonl(out / "baseline_val" / "predictions.jsonl", preds)
        (out / "baseline_val" / "metrics.json").write_text(json.dumps(report, indent=2))
        print("baseline:", headline(report), flush=True)

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    if not args.no_gradient_checkpointing:
        # Recompute layer activations in the backward pass instead of storing them.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    train_examples = [encode(tok, r, args.max_length) for r in train_records]
    val_batches = [
        collate([encode(tok, r, args.max_length) for r in val_records[i:i + args.eval_batch_size]],
                tok.pad_token_id)
        for i in range(0, len(val_records), args.eval_batch_size)
    ]

    steps_per_epoch = math.ceil(len(train_examples) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler(enabled=amp_dtype is not None)
    shuffler = random.Random(args.seed)

    history, val_by_epoch = [], []
    best_f1, step = -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        shuffler.shuffle(order)
        micro_batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]

        running = 0.0
        for mb_index, idx in enumerate(micro_batches):
            batch = collate([train_examples[i] for i in idx], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += float(loss)

            last = mb_index == len(micro_batches) - 1
            if (mb_index + 1) % args.grad_accum == 0 or last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                history.append({"step": step, "epoch": round(epoch - 1 + (mb_index + 1) / len(micro_batches), 3),
                                "loss": round(running, 5), "lr": scheduler.get_last_lr()[0],
                                "elapsed_s": round(time.perf_counter() - train_start, 1)})
                if step % 5 == 0 or last:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running:.4f}", flush=True)
                running = 0.0

        val_loss = validation_loss(model, val_batches, device, amp_dtype)
        model.eval()
        preds, report = evaluate(model, tok, val_records, batch_size=args.eval_batch_size)
        summary = {"epoch": epoch, "step": step, "val_loss": round(val_loss, 5), **headline(report)}
        val_by_epoch.append(summary)
        print("validation:", summary, flush=True)

        model.save_pretrained(out / f"epoch_{epoch}")
        if summary["macro_f1"] > best_f1:
            best_f1 = summary["macro_f1"]
            shutil.rmtree(out / "best", ignore_errors=True)
            model.save_pretrained(out / "best")
            write_jsonl(out / "best" / "val_predictions.jsonl", preds)
            (out / "best" / "val_metrics.json").write_text(json.dumps({**report, "epoch": epoch}, indent=2))

        (out / "history.json").write_text(json.dumps(history, indent=1))
        (out / "val_by_epoch.json").write_text(json.dumps(val_by_epoch, indent=2))

    resources = {"train_seconds": round(time.perf_counter() - train_start, 1),
                 "best_epoch": max(val_by_epoch, key=lambda s: s["macro_f1"])["epoch"]}
    if device.type == "cuda":
        resources["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "resources.json").write_text(json.dumps(resources, indent=2))
    print("done:", resources, flush=True)


if __name__ == "__main__":
    main()
