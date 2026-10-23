"""Run a model over dataset records and score the parsed predictions."""

import torch

from .metrics import full_report
from .modeling import generate_batch, prompt_text
from .prompting import parse_output


@torch.no_grad()
def predict(model, tok, records, batch_size=8, max_new_tokens=64, log_every=0, order="evidence_first"):
    predictions = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        outputs = generate_batch(model, tok, [prompt_text(tok, r, order) for r in batch], max_new_tokens)
        for record, text in zip(batch, outputs):
            label, evidence, ok = parse_output(
                text, record["criterion_type"], [sid for sid, _ in record["sentences"]]
            )
            predictions.append({
                "annotation_id": record["annotation_id"],
                "patient_id": record["patient_id"],
                "criterion_type": record["criterion_type"],
                "raw_output": text,
                "label": label,
                "evidence": evidence,
                "parse_ok": ok,
                "gold_label": record["label"],
                "gold_evidence": record["evidence"],
            })
        if log_every and (i // batch_size) % log_every == 0:
            print(f"  predicted {len(predictions)}/{len(records)}", flush=True)
    return predictions


def headline(report):
    """The scalar summary used for model selection and logs."""
    overall = report["overall"]
    return {
        "accuracy": overall["classification"]["accuracy"],
        "macro_f1": overall["classification"]["macro_f1"],
        "evidence_micro_f1": overall["evidence"]["micro"]["f1"],
        "parse_ok_rate": report["parse_ok_rate"],
    }


def evaluate(model, tok, records, **kwargs):
    predictions = predict(model, tok, records, **kwargs)
    return predictions, full_report(records, predictions)
