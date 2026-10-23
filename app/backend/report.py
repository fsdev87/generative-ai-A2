"""Evidence-grounded explanations, the explicit aggregation rule, and report export."""

import html
import time

DISCLAIMER = ("Educational screening prototype. Not a clinical enrollment decision: every "
              "result must be reviewed by qualified clinical staff.")

_DECISIVE = {"included", "not included", "excluded", "not excluded"}

_MEANING = {
    ("inclusion", "included"): "the patient appears to meet this inclusion criterion",
    ("inclusion", "not included"): "the patient appears not to meet this inclusion criterion",
    ("exclusion", "excluded"): "the patient appears to have the excluding condition",
    ("exclusion", "not excluded"): "the patient appears not to have the excluding condition",
}

AGGREGATION_RULES = [
    "Likely ineligible: any inclusion criterion is 'not included' or any exclusion criterion is 'excluded'.",
    "Needs clinician review: otherwise, if any criterion is 'not enough information' or could not be parsed.",
    "Potentially eligible: otherwise (every criterion is satisfied or 'not applicable').",
    "This verdict is a rule-based aggregation of criterion-level predictions, not a ground-truth label.",
]


def explain(ctype, ctext, label, evidence, parse_ok):
    """Build an explanation strictly from the cited sentences; return (text, flags)."""
    flags = []
    if label is None:
        return "The model output could not be parsed into a valid label.", ["unparseable"]
    if not parse_ok:
        flags.append("recovered_from_malformed_output")

    cited = " ".join(f"[{e['id']}] \"{e['text']}\"" for e in evidence)
    if label in _DECISIVE:
        if evidence:
            text = f"Based on {cited}, {_MEANING[(ctype, label)]}."
        else:
            text = (f"The model judged that {_MEANING[(ctype, label)]}, but cited no supporting "
                    "sentence; treat this as unsupported.")
            flags.append("unsupported")
    elif label == "not enough information":
        text = ("The note does not contain enough information to decide this criterion. Missing "
                f"information: whether the patient satisfies \"{ctext}\".")
        if evidence:
            text += f" Related but inconclusive sentences: {cited}."
        flags.append("missing_information")
    else:  # not applicable
        text = "The criterion does not apply to this patient."
        if evidence:
            text += f" Reason given by the cited sentences: {cited}."
    return text, flags


def aggregate(results):
    labels = [(r["criterion_type"], r["label"]) for r in results]
    blocking = [i for i, (t, l) in enumerate(labels)
                if (t, l) in {("inclusion", "not included"), ("exclusion", "excluded")}]
    unresolved = [i for i, (_, l) in enumerate(labels) if l in (None, "not enough information")]
    if blocking:
        verdict = "likely ineligible"
    elif unresolved:
        verdict = "needs clinician review"
    else:
        verdict = "potentially eligible"
    counts = {}
    for t, l in labels:
        counts.setdefault(t, {}).setdefault(l or "unparseable", 0)
        counts[t][l or "unparseable"] += 1
    return {"verdict": verdict, "blocking_criteria": blocking, "unresolved_criteria": unresolved,
            "counts": counts, "rules": AGGREGATION_RULES}


def render_html(report):
    """A standalone, printable HTML screening report."""
    e = html.escape
    rows = []
    for i, r in enumerate(report["results"]):
        evidence = "".join(f"<li><b>[{x['id']}]</b> {e(x['text'])}</li>" for x in r["evidence"]) or "<li>none cited</li>"
        flags = ", ".join(r["flags"]) or "&ndash;"
        rows.append(f"""
        <tr><td>{i + 1}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion_text'])}</td>
        <td><b>{e(r['label'] or 'unparseable')}</b></td><td><ul>{evidence}</ul></td>
        <td>{e(r['explanation'])}</td><td>{flags}</td></tr>""")
    note = "".join(f"<li value='{sid}'>{e(text)}</li>" for sid, text in report["sentences"])
    prov = "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in report["provenance"].items())
    rules = "".join(f"<li>{e(rule)}</li>" for rule in report["aggregate"]["rules"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Screening report {e(report['id'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:32px;color:#1d2433;line-height:1.45}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #c9d1de;padding:6px 8px;vertical-align:top;text-align:left;font-size:14px}}
th{{background:#eef2f8}}.warn{{background:#fff4d6;border:1px solid #e0b84c;padding:10px 14px;border-radius:6px}}
ol{{padding-left:28px}}ul{{margin:0;padding-left:18px}}h1{{margin-bottom:4px}}</style></head><body>
<h1>Eligibility screening report</h1>
<p>Report {e(report['id'])} &middot; created {e(report['created_at'])}{(' &middot; trial ' + e(report['trial_id'])) if report.get('trial_id') else ''}</p>
<p class="warn"><b>Disclaimer:</b> {e(DISCLAIMER)}</p>
<h2>Preliminary outcome: {e(report['aggregate']['verdict'])}</h2><ol>{rules}</ol>
<h2>Criterion-level results</h2>
<table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Prediction</th><th>Evidence</th><th>Explanation</th><th>Flags</th></tr>{''.join(rows)}</table>
<h2>Patient note (numbered sentences)</h2><ol start="0">{note}</ol>
<h2>Provenance</h2><table>{prov}</table>
</body></html>"""


def new_report(report_id, sentences, results, provenance, trial_id=None, trial_title=None):
    return {"id": report_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trial_id": trial_id, "trial_title": trial_title, "disclaimer": DISCLAIMER,
            "sentences": sentences, "results": results, "aggregate": aggregate(results),
            "provenance": provenance}
