"""Metrics checked against hand-computed values."""

import pytest

from trialmatch.metrics import INVALID, classification_report, evidence_report, full_report


def test_classification_hand_computed():
    gold = ["a", "a", "a", "b", "b", "c"]
    pred = ["a", "a", "b", "b", None, "a"]
    r = classification_report(gold, pred)
    assert r["accuracy"] == pytest.approx(3 / 6, abs=1e-4)
    # a: P=2/3 R=2/3 F1=2/3; b: P=1/2 R=1/2 F1=1/2; c: P=0 R=0 F1=0
    assert r["per_class"]["a"]["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["per_class"]["b"]["f1"] == pytest.approx(0.5, abs=1e-4)
    assert r["per_class"]["c"]["f1"] == 0
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0) / 3, abs=1e-4)
    assert r["invalid_predictions"] == 1
    labels = r["confusion_matrix"]["labels"]
    assert labels == ["a", "b", "c", INVALID]
    assert r["confusion_matrix"]["matrix"][1] == [0, 1, 0, 1]  # gold b: one b, one invalid


def test_macro_ignores_classes_absent_from_gold():
    r = classification_report(["a", "a"], ["a", "z"])
    assert set(r["per_class"]) == {"a"}
    assert r["macro_f1"] == pytest.approx(2 / 3, abs=1e-4)  # a: P=1, R=0.5


def test_evidence_hand_computed():
    gold = [[1, 2], [], [3], []]
    pred = [[2, 4], [5], [], []]
    r = evidence_report(gold, pred)
    # TP=1 (2); FP=3 (4, 5, ...) -> 4 and 5 = 2; FN=2 (1, 3)
    assert r["micro"]["precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["micro"]["recall"] == pytest.approx(1 / 3, abs=1e-4)
    # macro over non-empty gold: record0 P=.5 R=.5 F1=.5; record2 P=0 R=0 F1=0
    assert r["macro_nonempty_gold"]["f1"] == pytest.approx(0.25, abs=1e-4)
    assert r["macro_nonempty_gold"]["n"] == 2
    assert r["empty_gold"] == {"n": 2, "agreement": 0.5}


def test_full_report_splits_by_type():
    records = [{"criterion_type": "inclusion", "label": "included", "evidence": [0]},
               {"criterion_type": "exclusion", "label": "not excluded", "evidence": []}]
    preds = [{"label": "included", "evidence": [0], "parse_ok": True},
             {"label": "excluded", "evidence": [], "parse_ok": True}]
    r = full_report(records, preds)
    assert r["inclusion"]["classification"]["accuracy"] == 1.0
    assert r["exclusion"]["classification"]["accuracy"] == 0.0
    assert r["overall"]["classification"]["accuracy"] == 0.5
