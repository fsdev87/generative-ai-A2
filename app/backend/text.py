"""Turning free-text user input into numbered sentences and individual criteria."""

import json
import re
import urllib.error
import urllib.request

from trialmatch.data import split_note

_NUMBERED = re.compile(r"^\s*\d+\.\s+")
# Sentence end: . ! ? followed by whitespace and an uppercase letter, digit or quote.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Do not split after common clinical abbreviations ("Dr. Smith", "approx. 5").
_ABBREVIATIONS = ("dr.", "mr.", "mrs.", "ms.", "approx.", "e.g.", "i.e.", "vs.", "no.", "st.", "fig.")
_BULLET = re.compile(r"^\s*(?:[-*•●]|\(?\d+[.)]|\(?[a-z][.)])\s+", re.IGNORECASE)


def split_sentences(note):
    """Return [[id, sentence], ...] with IDs starting at 0.

    Notes already in the dataset's "0. text" format keep their numbering; anything
    else is split on line breaks and sentence boundaries.
    """
    lines = [l for l in note.strip().splitlines() if l.strip()]
    if lines and all(_NUMBERED.match(l) for l in lines[:2]) and _NUMBERED.match(lines[0]):
        try:
            return split_note(note.strip())
        except ValueError:
            pass

    sentences = []
    for line in lines:
        pieces = _SENTENCE_END.split(line.strip())
        merged = []
        for piece in pieces:
            if merged and merged[-1].lower().endswith(_ABBREVIATIONS):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        sentences.extend(p.strip() for p in merged if p.strip())
    return [[i, s] for i, s in enumerate(sentences)]


def split_criteria(block):
    """Split a pasted criteria block (bullets, numbering or one per line) into items."""
    items = []
    for line in block.splitlines():
        if not line.strip():
            continue
        if _BULLET.match(line) or not items:
            items.append(_BULLET.sub("", line).strip())
        elif line.startswith((" ", "\t")):
            items[-1] += " " + line.strip()   # indented continuation of the previous item
        else:
            items.append(line.strip())
    return [i for i in items if i and not i.lower().rstrip(":").endswith("criteria")]


def fetch_trial(nct_id, timeout=10):
    """Fetch a trial's title and eligibility criteria from the public ClinicalTrials.gov API."""
    nct_id = nct_id.strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise ValueError("NCT ID must look like NCT01234567")
    url = (f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
           "?fields=protocolSection.identificationModule,protocolSection.eligibilityModule")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise LookupError(f"ClinicalTrials.gov returned HTTP {e.code} for {nct_id}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ConnectionError("Could not reach ClinicalTrials.gov (offline?)") from e

    section = data.get("protocolSection", {})
    title = section.get("identificationModule", {}).get("briefTitle", "")
    text = section.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    inclusion, exclusion = _split_inclusion_exclusion(text)
    return {"nct_id": nct_id, "title": title, "inclusion": inclusion, "exclusion": exclusion,
            "source": url}


def _split_inclusion_exclusion(text):
    match = re.search(r"exclusion criteria\s*:?", text, flags=re.IGNORECASE)
    inc_text, exc_text = (text[:match.start()], text[match.end():]) if match else (text, "")
    inc_text = re.sub(r"^\s*inclusion criteria\s*:?", "", inc_text, flags=re.IGNORECASE)
    return split_criteria(inc_text), split_criteria(exc_text)
