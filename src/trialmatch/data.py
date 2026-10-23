"""Dataset loading, the prescribed patient-grouped split, and record construction."""

import json
import re

import numpy as np

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
SPLIT_SEED = 42
SPLIT_SIZES = {"train": 37, "val": 8, "test": 8}

# Columns produced by GPT-4, judgements about GPT-4 output, or the original authors'
# own train flag. They are dropped at load time so they can never reach a prompt,
# a target or an evaluation.
LEAKY_COLUMNS = [
    "gpt4_explanation",
    "gpt4_sentences",
    "gpt4_eligibility",
    "explanation_correctness",
    "training",
]

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}

_SENTENCE_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def load_annotations(cache_dir=None):
    """Load all 1,015 annotations with the leaky columns removed."""
    from datasets import load_dataset  # not needed (or installed) in the app image

    ds = load_dataset(DATASET_ID, split="train", cache_dir=cache_dir)
    ds = ds.remove_columns(LEAKY_COLUMNS)
    return [dict(row) for row in ds]


def split_note(note):
    """Parse a dataset note ("0. text\\n1. text ...") into [(sentence_id, text), ...].

    Lines without a leading number are treated as continuations of the previous
    sentence so that no patient text is ever dropped.
    """
    sentences = []
    for line in note.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _SENTENCE_RE.match(line)
        if match:
            sentences.append([int(match[1]), match[2]])
        elif sentences:
            sentences[-1][1] += " " + line
        else:
            raise ValueError(f"Note does not start with a numbered sentence: {line[:80]!r}")
    return sentences


def make_record(row):
    """Convert a raw dataset row into the model-facing record."""
    sentences = split_note(row["note"])
    valid_ids = {sid for sid, _ in sentences}
    evidence = sorted(set(json.loads(row["expert_sentences"] or "[]")))
    label = row["expert_eligibility"]
    ctype = row["criterion_type"]

    if label not in LABELS[ctype]:
        raise ValueError(f"annotation {row['annotation_id']}: label {label!r} invalid for {ctype}")
    if not set(evidence) <= valid_ids:
        raise ValueError(f"annotation {row['annotation_id']}: evidence {evidence} outside note ids")

    return {
        "annotation_id": row["annotation_id"],
        "patient_id": row["patient_id"],
        "trial_id": row["trial_id"],
        "criterion_type": ctype,
        "criterion_text": row["criterion_text"],
        "sentences": sentences,
        "label": label,
        "evidence": evidence,
    }


def patient_split(patient_ids):
    """The prescribed split: sort unique IDs, shuffle with default_rng(42), take 37/8/8."""
    ids = np.array(sorted(set(patient_ids)))
    expected = sum(SPLIT_SIZES.values())
    if len(ids) != expected:
        raise ValueError(f"expected {expected} unique patients, found {len(ids)}")

    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ids)

    n_train, n_val = SPLIT_SIZES["train"], SPLIT_SIZES["val"]
    return {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train:n_train + n_val].tolist(),
        "test": ids[n_train + n_val:].tolist(),
    }


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
