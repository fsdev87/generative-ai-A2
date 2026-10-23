"""Split integrity, leakage and prompt/target construction checks on the processed data."""

import json
from pathlib import Path

import pytest

from trialmatch.data import LABELS, LEAKY_COLUMNS, SPLIT_SIZES, patient_split, read_jsonl
from trialmatch.prompting import ANSWER_ORDERS, build_messages, build_target, parse_output

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {name: read_jsonl(ROOT / "data" / "processed" / f"{name}.jsonl") for name in SPLIT_SIZES}
MANIFEST = json.loads((ROOT / "data" / "splits" / "split_manifest.json").read_text())


def test_split_is_reproducible_from_the_protocol():
    all_patients = MANIFEST["splits"]["train"]["patient_ids"] + MANIFEST["splits"]["val"]["patient_ids"] \
        + MANIFEST["splits"]["test"]["patient_ids"]
    recomputed = patient_split(all_patients)
    for name in SPLIT_SIZES:
        assert recomputed[name] == MANIFEST["splits"][name]["patient_ids"]
        assert len(recomputed[name]) == SPLIT_SIZES[name]


def test_no_patient_in_two_splits_and_records_match_manifest():
    sets = {name: {r["patient_id"] for r in recs} for name, recs in SPLITS.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    for name in SPLIT_SIZES:
        assert sets[name] == set(MANIFEST["splits"][name]["patient_ids"])


def test_every_annotation_once_except_documented_drops():
    ids = [r["annotation_id"] for recs in SPLITS.values() for r in recs]
    dropped = {d["annotation_id"] for d in MANIFEST["dropped_annotations"]}
    assert len(ids) == len(set(ids))
    assert len(ids) + len(dropped) == 1015
    assert not set(ids) & dropped


def test_no_leaky_fields_in_processed_records():
    for recs in SPLITS.values():
        for r in recs:
            assert not set(LEAKY_COLUMNS) & set(r)
            assert not any(k.startswith("gpt4") for k in r)


def test_labels_and_evidence_are_valid():
    for recs in SPLITS.values():
        for r in recs:
            assert r["label"] in LABELS[r["criterion_type"]]
            assert set(r["evidence"]) <= {sid for sid, _ in r["sentences"]}
            assert r["criterion_text"].strip()


@pytest.mark.parametrize("order", ANSWER_ORDERS)
def test_prompt_does_not_depend_on_the_answer(order):
    """The prompt must be identical whatever the gold label/evidence is: no leakage path."""
    for r in SPLITS["val"][:50]:
        prompt = build_messages(r, order)
        for other_label in LABELS[r["criterion_type"]]:
            altered = {**r, "label": other_label, "evidence": [0]}
            assert build_messages(altered, order) == prompt


@pytest.mark.parametrize("order", ANSWER_ORDERS)
def test_target_round_trips_through_the_parser(order):
    for recs in SPLITS.values():
        for r in recs:
            label, evidence, ok = parse_output(build_target(r, order), r["criterion_type"],
                                               [sid for sid, _ in r["sentences"]])
            assert ok and label == r["label"] and evidence == r["evidence"]


def test_parser_rejects_invalid_and_cleans_ids():
    ids = [0, 1, 2]
    assert parse_output('{"label": "excluded", "evidence": [1]}', "inclusion", ids)[0] is None
    assert parse_output('{"label": "included", "evidence": [1, 9, "x"]}', "inclusion", ids) == ("included", [1], True)
    assert parse_output("the patient is not included", "inclusion", ids) == ("not included", [], False)
    assert parse_output("garbage", "exclusion", ids) == (None, [], False)
