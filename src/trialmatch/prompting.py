"""Prompt construction, training targets, and robust parsing of model output.

Output schema (one JSON object). The default puts evidence first so the model commits
to the cited sentences before the label; "label_first" is the ablation:

    evidence_first: {"evidence": [<sentence ids>], "label": "<label>"}
    label_first:    {"label": "<label>", "evidence": [<sentence ids>]}
"""

import json
import re

from .data import LABELS

SYSTEM_PROMPT = (
    "You are a clinical trial screening assistant. You judge whether a patient "
    "satisfies one eligibility criterion of a clinical trial, using only the "
    "information in the patient note. Missing information is never proof of "
    "eligibility or ineligibility."
)

LABEL_DEFINITIONS = {
    "inclusion": {
        "included": "the note shows the patient meets the criterion",
        "not included": "the note shows the patient does not meet the criterion",
        "not enough information": "the note does not contain enough information to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
    "exclusion": {
        "excluded": "the note shows the patient has the excluding condition",
        "not excluded": "the note shows the patient does not have the excluding condition",
        "not enough information": "the note does not contain enough information to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
}


ANSWER_ORDERS = ("evidence_first", "label_first")
_SCHEMAS = {
    "evidence_first": '{"evidence": [<sentence ids>], "label": "<label>"}',
    "label_first": '{"label": "<label>", "evidence": [<sentence ids>]}',
}


def format_note(sentences):
    return "\n".join(f"{sid}. {text}" for sid, text in sentences)


def build_user_prompt(sentences, criterion_type, criterion_text, order="evidence_first"):
    definitions = "\n".join(
        f'- "{label}": {meaning}' for label, meaning in LABEL_DEFINITIONS[criterion_type].items()
    )
    return (
        "Patient note (sentences are numbered):\n"
        f"{format_note(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Labels:\n{definitions}\n\n"
        "Cite the IDs of the patient sentences that support your label "
        "(an empty list if no sentence is relevant).\n"
        f"Answer with JSON only: {_SCHEMAS[order]}"
    )


def build_messages(record, order="evidence_first"):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                record["sentences"], record["criterion_type"], record["criterion_text"], order
            ),
        },
    ]


def build_target(record, order="evidence_first"):
    if order == "label_first":
        return json.dumps({"label": record["label"], "evidence": record["evidence"]})
    return json.dumps({"evidence": record["evidence"], "label": record["label"]})


def parse_output(text, criterion_type, valid_ids):
    """Parse generated text into (label, evidence, parse_ok).

    label is None when no valid label can be recovered. Evidence IDs that do not
    exist in the note are dropped. parse_ok is True only for well-formed JSON with
    a valid label.
    """
    allowed = LABELS[criterion_type]
    valid_ids = set(valid_ids)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = str(obj.get("label", "")).strip().lower().replace("_", " ")
            evidence = obj.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            evidence = sorted({int(e) for e in evidence if str(e).lstrip("-").isdigit()} & valid_ids)
            if label in allowed:
                return label, evidence, True
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass

    # Fallback: longest allowed label mentioned in the text ("not included" before "included").
    lowered = text.lower().replace("_", " ")
    for label in sorted(allowed, key=len, reverse=True):
        if label in lowered:
            return label, [], False
    return None, [], False
