"""Evaluate a base or fine-tuned model on the val (or, with --final, test) split.

Outputs in --out:
  predictions.jsonl   one line per annotation: raw generation, parsed label/evidence, gold
  metrics.json        classification + evidence metrics, model and decoding settings

Usage:
  python scripts/evaluate.py --split val --out results/zeroshot_val
  python scripts/evaluate.py --split val --adapter checkpoints/lora --out results/lora_val
  python scripts/evaluate.py --split test --final --adapter ... --out results/lora_test
"""

import argparse
import json
import time
from pathlib import Path

from trialmatch.data import read_jsonl, write_jsonl
from trialmatch.evaluation import evaluate, headline
from trialmatch.modeling import TARGET_MODEL, load_model, load_tokenizer
from trialmatch.prompting import ANSWER_ORDERS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true",
                        help="required to evaluate on the held-out test split")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--answer-order", choices=ANSWER_ORDERS, default="evidence_first",
                        help="must match the order the adapter was trained with")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        parser.error("the test split is held out; pass --final only for the final evaluation")

    records = read_jsonl(ROOT / "data" / "processed" / f"{args.split}.jsonl")[: args.limit]
    tok = load_tokenizer(args.model)
    model = load_model(args.model, adapter=args.adapter)

    start = time.perf_counter()
    predictions, report = evaluate(model, tok, records, batch_size=args.batch_size,
                                   max_new_tokens=args.max_new_tokens, log_every=5,
                                   order=args.answer_order)
    report["settings"] = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n": len(records),
        "answer_order": args.answer_order,
        "decoding": {"strategy": "greedy", "do_sample": False, "repetition_penalty": 1.0,
                     "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
        "device": str(model.device),
        "dtype": str(model.dtype),
        "seconds": round(time.perf_counter() - start, 1),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", predictions)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(headline(report))


if __name__ == "__main__":
    main()
