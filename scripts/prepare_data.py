"""Build the prescribed 37/8/8 patient split, processed JSONL files, and token statistics.

Outputs:
  data/splits/split_manifest.json   patient IDs, counts and label distribution per split
  data/processed/{train,val,test}.jsonl
  data/processed/token_stats.json   prompt+target lengths and truncation counts

Usage:
  python scripts/prepare_data.py [--tokenizer Qwen/Qwen2.5-1.5B-Instruct]
"""

import argparse
import collections
import json
from pathlib import Path

from trialmatch.data import (
    DATASET_ID,
    LEAKY_COLUMNS,
    SPLIT_SEED,
    load_annotations,
    make_record,
    patient_split,
    write_jsonl,
)
from trialmatch.prompting import build_messages, build_target

ROOT = Path(__file__).resolve().parents[1]
TRUNCATION_LIMITS = [256, 384, 512, 768, 1024]


def check_split(splits, records_by_split, all_records):
    """Leakage and integrity checks; raises on any violation."""
    sets = {name: set(ids) for name, ids in splits.items()}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise AssertionError(f"patients shared between {a} and {b}: {sets[a] & sets[b]}")

    seen = collections.Counter(r["annotation_id"] for rs in records_by_split.values() for r in rs)
    if len(seen) != len(all_records) or any(c != 1 for c in seen.values()):
        raise AssertionError("every annotation must appear in exactly one split")

    for name, records in records_by_split.items():
        if {r["patient_id"] for r in records} != sets[name]:
            raise AssertionError(f"{name}: records do not match the patients assigned to it")
        for r in records:
            leaked = set(LEAKY_COLUMNS) & set(r)
            if leaked:
                raise AssertionError(f"leaky fields {leaked} present in annotation {r['annotation_id']}")
    return {
        "patient_overlap": "none",
        "annotations_assigned_once": True,
        "leaky_fields_in_records": "none",
    }


def summarize(records):
    by_type = collections.defaultdict(collections.Counter)
    for r in records:
        by_type[r["criterion_type"]][r["label"]] += 1
    return {
        "annotations": len(records),
        "with_evidence": sum(bool(r["evidence"]) for r in records),
        "without_evidence": sum(not r["evidence"] for r in records),
        "labels": {ctype: dict(sorted(c.items())) for ctype, c in sorted(by_type.items())},
    }


def token_stats(records_by_split, tokenizer_name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    stats = {"tokenizer": tokenizer_name, "splits": {}}
    for name, records in records_by_split.items():
        prompt_lens, full_lens, target_lens = [], [], []
        for r in records:
            prompt_text = tok.apply_chat_template(
                build_messages(r), add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tok(build_target(r) + tok.eos_token, add_special_tokens=False)["input_ids"]
            prompt_lens.append(len(prompt_ids))
            target_lens.append(len(target_ids))
            full_lens.append(len(prompt_ids) + len(target_ids))
        stats["splits"][name] = {
            "prompt_tokens": {"min": min(prompt_lens), "max": max(prompt_lens),
                              "mean": round(sum(prompt_lens) / len(prompt_lens), 1)},
            "target_tokens": {"min": min(target_lens), "max": max(target_lens),
                              "mean": round(sum(target_lens) / len(target_lens), 1)},
            "prompt_plus_target_max": max(full_lens),
            "examples_exceeding": {str(n): sum(l > n for l in full_lens) for n in TRUNCATION_LIMITS},
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--skip-tokens", action="store_true")
    args = parser.parse_args()

    rows = load_annotations(cache_dir=ROOT / "data" / "raw")
    records = [make_record(row) for row in rows]
    splits = patient_split(r["patient_id"] for r in records)

    records_by_split = {
        name: sorted((r for r in records if r["patient_id"] in set(ids)),
                     key=lambda r: r["annotation_id"])
        for name, ids in splits.items()
    }
    checks = check_split(splits, records_by_split, records)

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, recs in records_by_split.items():
        write_jsonl(processed / f"{name}.jsonl", recs)

    manifest = {
        "dataset": DATASET_ID,
        "procedure": "sorted unique patient_id -> numpy.random.default_rng(42).shuffle -> "
                     "first 37 train, next 8 val, last 8 test",
        "seed": SPLIT_SEED,
        "excluded_columns": LEAKY_COLUMNS,
        "checks": checks,
        "splits": {
            name: {"patients": len(ids), "patient_ids": ids, **summarize(records_by_split[name])}
            for name, ids in splits.items()
        },
        "total": summarize(records),
    }
    split_dir = ROOT / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    for name in splits:
        s = manifest["splits"][name]
        print(f"{name:5s} patients={s['patients']:2d} annotations={s['annotations']:4d} labels={s['labels']}")
    print("checks:", checks)

    if not args.skip_tokens:
        stats = token_stats(records_by_split, args.tokenizer)
        (processed / "token_stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
