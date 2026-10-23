"""API tests with the small 0.5B model standing in for the fine-tuned target."""

import json
import os

import pytest

os.environ["SKIP_MODEL_LOAD"] = "1"
os.environ["TARGET_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["MAX_NEW_TOKENS"] = "24"
os.environ["MAX_PROMPT_TOKENS"] = "1024"

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import app, engine  # noqa: E402
from app.backend.report import aggregate  # noqa: E402
from app.backend.text import split_criteria, split_sentences  # noqa: E402

NOTE = ("A 58-year-old woman presents with chest pain. She has hypertension and obesity. "
        "She denies smoking or diabetes. She takes no medications.")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import app.backend.main as main

    main.REPORTS = tmp_path_factory.mktemp("reports")
    engine.settings.__init__()
    engine.load()
    assert engine.state == "ready", engine.error
    with TestClient(app) as c:
        yield c


def test_sentence_and_criteria_parsing():
    assert [s for _, s in split_sentences(NOTE)][:2] == ["A 58-year-old woman presents with chest pain.",
                                                       "She has hypertension and obesity."]
    assert split_sentences("0. First.\n1. Second.") == [[0, "First."], [1, "Second."]]
    block = "Inclusion Criteria:\n\n* Age 18 or older\n* Diagnosed with\n  stable angina\n- Signed consent"
    assert split_criteria(block) == ["Age 18 or older", "Diagnosed with stable angina", "Signed consent"]


def test_aggregation_rules():
    r = lambda t, l: {"criterion_type": t, "label": l}
    assert aggregate([r("inclusion", "included"), r("exclusion", "excluded")])["verdict"] == "likely ineligible"
    assert aggregate([r("inclusion", "included"), r("exclusion", "not enough information")])["verdict"] \
        == "needs clinician review"
    assert aggregate([r("inclusion", "included"), r("exclusion", "not applicable")])["verdict"] \
        == "potentially eligible"


def test_health(client):
    body = client.get("/api/health").json()
    assert body["state"] == "ready" and "speculative" in body["modes"]


def test_screen_stream_and_report(client):
    resp = client.post("/api/screen", json={
        "note": NOTE, "trial_id": "NCT00000000",
        "criteria": [{"type": "inclusion", "text": "Age over 50"},
                     {"type": "exclusion", "text": "Current smoker"}]})
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines()]
    assert [e["event"] for e in events] == ["start", "result", "result", "done"]
    for e in events[1:3]:
        assert "label" in e["result"] and "timing" in e["result"]
    report_id = events[-1]["report_id"]
    assert client.get(f"/api/reports/{report_id}").json()["provenance"]["target_model"]
    html = client.get(f"/api/reports/{report_id}/download?fmt=html").text
    assert "Educational screening prototype" in html
    assert any(r["id"] == report_id for r in client.get("/api/reports").json())


def test_second_criterion_reuses_prefix(client):
    client.post("/api/screen", json={"note": NOTE, "criteria": [{"type": "inclusion", "text": "Age over 50"}]})
    resp = client.post("/api/screen", json={"note": NOTE, "criteria": [{"type": "inclusion", "text": "Female"}]})
    result = [json.loads(l) for l in resp.text.splitlines()][1]["result"]
    assert result["timing"]["reused_prefix_tokens"] > 50


@pytest.mark.parametrize("payload,status", [
    ({"note": "", "criteria": [{"type": "inclusion", "text": "Age over 50"}]}, 422),
    ({"note": NOTE, "criteria": []}, 422),
    ({"note": NOTE, "criteria": [{"type": "maybe", "text": "Age over 50"}]}, 422),
    ({"note": NOTE, "criteria": [{"type": "inclusion", "text": "   "}]}, 422),
    ({"note": NOTE, "criteria": [{"type": "inclusion", "text": "Age"}], "mode": "turbo"}, 422),
    ({"note": NOTE * 60, "criteria": [{"type": "inclusion", "text": "Age over 50"}]}, 422),  # too many tokens
])
def test_invalid_inputs_are_rejected(client, payload, status):
    resp = client.post("/api/screen", json=payload)
    assert resp.status_code == status
    assert "detail" in resp.json()


def test_benchmark_modes_agree(client):
    resp = client.post("/api/benchmark", json={
        "note": NOTE, "criterion_type": "inclusion", "criterion_text": "Age over 50",
        "modes": ["cached", "uncached", "cached_tiled"], "max_new_tokens": 8})
    rows = resp.json()["rows"]
    assert [r["mode"] for r in rows] == ["cached", "uncached", "cached_tiled"]
    assert all(r["tokens_equal_first_mode"] for r in rows)


def test_attention_check(client):
    rows = client.post("/api/attention-check", json={"seq_len": 64, "block": 16}).json()["rows"]
    assert all(r["max_abs_err"] < 1e-4 for r in rows)


def test_bad_trial_id(client):
    assert client.get("/api/trial/not-an-id").status_code == 422


def test_examples_come_from_validation_split(client):
    examples = client.get("/api/examples?limit=3").json()
    assert len(examples) == 3 and all(e["split"] == "val" for e in examples)
