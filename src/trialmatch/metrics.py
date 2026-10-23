"""Criterion classification and evidence-retrieval metrics.

Classification
  * Predictions that cannot be parsed into a valid label are mapped to "invalid"
    and always count as errors.
  * Macro-F1 averages over the gold classes present in the evaluated subset
    (classes with zero gold support have undefined recall and are reported
    separately rather than silently scored as 0 or 1).

Evidence (sentence IDs)
  * micro P/R/F1: TP/FP/FN summed over every example. Examples whose gold set is
    empty contribute their predicted IDs as false positives.
  * macro P/R/F1: per-example scores averaged over examples with a non-empty gold
    set (P is taken as 0 when the prediction is empty).
  * empty_gold_agreement: fraction of empty-gold examples where the model also
    cited nothing.
"""

from collections import Counter

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

INVALID = "invalid"


def _safe_div(a, b):
    return a / b if b else 0.0


def classification_report(gold, pred):
    pred = [p if p is not None else INVALID for p in pred]
    support = Counter(gold)
    classes = sorted(support)
    matrix_labels = classes + sorted(set(pred) - set(classes))

    p, r, f, _ = precision_recall_fscore_support(
        gold, pred, labels=classes, zero_division=0
    )
    per_class = {
        c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4), "support": support[c]}
        for i, c in enumerate(classes)
    }
    return {
        "n": len(gold),
        "accuracy": round(sum(g == q for g, q in zip(gold, pred)) / len(gold), 4) if gold else 0.0,
        "macro_f1": round(float(f.mean()), 4) if len(classes) else 0.0,
        "invalid_predictions": pred.count(INVALID),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": matrix_labels,
            "matrix": confusion_matrix(gold, pred, labels=matrix_labels).tolist(),
        },
    }


def evidence_report(gold_sets, pred_sets):
    tp = fp = fn = 0
    precisions, recalls, f1s = [], [], []
    empty_gold = empty_agree = 0

    for gold, pred in zip(gold_sets, pred_sets):
        gold, pred = set(gold), set(pred)
        hit = len(gold & pred)
        tp += hit
        fp += len(pred - gold)
        fn += len(gold - pred)

        if not gold:
            empty_gold += 1
            empty_agree += not pred
            continue
        prec = _safe_div(hit, len(pred))
        rec = hit / len(gold)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(_safe_div(2 * prec * rec, prec + rec))

    micro_p, micro_r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4),
                  "f1": round(_safe_div(2 * micro_p * micro_r, micro_p + micro_r), 4)},
        "macro_nonempty_gold": {
            "precision": round(_safe_div(sum(precisions), len(precisions)), 4),
            "recall": round(_safe_div(sum(recalls), len(recalls)), 4),
            "f1": round(_safe_div(sum(f1s), len(f1s)), 4),
            "n": len(f1s),
        },
        "empty_gold": {"n": empty_gold,
                       "agreement": round(_safe_div(empty_agree, empty_gold), 4)},
    }


def full_report(records, predictions):
    """records: gold dataset records; predictions: dicts with 'label' and 'evidence'."""
    report = {}
    subsets = {"overall": None, "inclusion": "inclusion", "exclusion": "exclusion"}
    for name, ctype in subsets.items():
        pairs = [(r, p) for r, p in zip(records, predictions)
                 if ctype is None or r["criterion_type"] == ctype]
        if not pairs:
            report[name] = None
            continue
        gold = [r["label"] for r, _ in pairs]
        pred = [p["label"] for _, p in pairs]
        report[name] = {
            "classification": classification_report(gold, pred),
            "evidence": evidence_report([r["evidence"] for r, _ in pairs],
                                        [p["evidence"] for _, p in pairs]),
        }
    report["parse_ok_rate"] = round(
        sum(p.get("parse_ok", False) for p in predictions) / len(predictions), 4
    )
    return report
