"""FastAPI server: screening API, live inference benchmarks, reports, and the React UI.

Run:  uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import threading
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from trialmatch.data import read_jsonl

from .engine import MODES, Engine, InputTooLong
from .report import new_report, render_html
from .text import fetch_trial, split_criteria, split_sentences

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = Path(os.getenv("FRONTEND_DIST", ROOT / "frontend" / "dist"))
REPORTS = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
RESULTS = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
Mode = Literal["cached", "cached_tiled", "uncached", "speculative"]

engine = Engine()


@asynccontextmanager
async def lifespan(_app):
    REPORTS.mkdir(parents=True, exist_ok=True)
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        threading.Thread(target=engine.load, daemon=True).start()  # UI is usable while loading
    yield


app = FastAPI(title="Trial Criterion Screening", lifespan=lifespan)


@app.exception_handler(InputTooLong)
async def too_long(_req: Request, exc: InputTooLong):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(MemoryError)
async def out_of_memory(_req: Request, exc: MemoryError):
    return JSONResponse(status_code=503, content={"detail": str(exc) or "Not enough memory."})


def require_ready():
    if engine.state == "loading":
        raise HTTPException(503, "The model is still loading. Please wait a moment and retry.")
    if engine.state == "error":
        raise HTTPException(503, f"The model failed to load: {engine.error}")


# --- request models -----------------------------------------------------------

class Criterion(BaseModel):
    type: Literal["inclusion", "exclusion"]
    text: str = Field(min_length=3, max_length=2000)

    @field_validator("text")
    @classmethod
    def strip(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("criterion text is empty")
        return v.strip()


class NoteIn(BaseModel):
    note: str = Field(min_length=10, max_length=20000)

    @field_validator("note")
    @classmethod
    def has_text(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("the patient note is empty or too short")
        return v


class ScreenIn(NoteIn):
    criteria: list[Criterion] = Field(min_length=1, max_length=60)
    mode: Mode = "cached"
    trial_id: str | None = Field(default=None, max_length=32)
    trial_title: str | None = Field(default=None, max_length=500)


class BenchmarkIn(NoteIn):
    criterion_type: Literal["inclusion", "exclusion"]
    criterion_text: str = Field(min_length=3, max_length=2000)
    modes: list[Mode] = Field(min_length=1, max_length=4)
    max_new_tokens: int = Field(default=48, ge=1, le=256)
    stop_at_eos: bool = False


class AttentionIn(BaseModel):
    seq_len: int = Field(default=512, ge=8, le=2048)
    causal: bool = True
    block: int = Field(default=64, ge=8, le=512)


class TextIn(BaseModel):
    text: str = Field(max_length=50000)


# --- endpoints ----------------------------------------------------------------

@app.get("/api/health")
def health():
    return engine.status()


@app.post("/api/sentences")
def sentences(body: NoteIn):
    return {"sentences": split_sentences(body.note)}


@app.post("/api/criteria/parse")
def parse_criteria(body: TextIn):
    return {"items": split_criteria(body.text)}


@app.get("/api/trial/{nct_id}")
def trial(nct_id: str):
    try:
        return fetch_trial(nct_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ConnectionError as e:
        raise HTTPException(502, str(e))


@app.get("/api/examples")
def examples(limit: int = 8):
    """Patient/trial pairs from the validation split (never the held-out test split)."""
    records = read_jsonl(ROOT / "data" / "processed" / "val.jsonl")
    groups = defaultdict(list)
    for r in records:
        groups[(r["patient_id"], r["trial_id"])].append(r)
    out = []
    for (patient, trial_id), rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:limit]:
        out.append({
            "patient_id": patient, "trial_id": trial_id, "split": "val",
            "note": "\n".join(f"{sid}. {text}" for sid, text in rs[0]["sentences"]),
            "criteria": [{"type": r["criterion_type"], "text": r["criterion_text"],
                          "expert_label": r["label"], "expert_evidence": r["evidence"]} for r in rs],
        })
    return out


@app.post("/api/screen")
def screen(body: ScreenIn):
    require_ready()
    sents = split_sentences(body.note)
    if not sents:
        raise HTTPException(422, "No sentences could be extracted from the patient note.")
    for c in body.criteria:  # validate every prompt length before starting the stream
        engine.prompt_ids(sents, c.type, c.text)

    def events():
        yield json.dumps({"event": "start", "sentences": sents, "total": len(body.criteria)}) + "\n"
        results = []
        for i, c in enumerate(body.criteria):
            try:
                result = engine.screen_one(sents, c.type, c.text, body.mode)
            except Exception as e:  # one failing criterion must not abort the whole screening
                result = {"criterion_type": c.type, "criterion_text": c.text, "label": None,
                          "evidence": [], "explanation": f"Inference failed: {e}",
                          "flags": ["error"], "raw_output": "", "mode": body.mode, "timing": {}}
            results.append(result)
            yield json.dumps({"event": "result", "index": i, "result": result}) + "\n"
        report_id = uuid.uuid4().hex[:10]
        report = new_report(report_id, sents, results, engine.provenance(body.mode),
                            body.trial_id, body.trial_title)
        (REPORTS / f"{report_id}.json").write_text(json.dumps(report, indent=2))
        (REPORTS / f"{report_id}.html").write_text(render_html(report), encoding="utf-8")
        yield json.dumps({"event": "done", "report_id": report_id, "aggregate": report["aggregate"]}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/reports")
def list_reports():
    items = []
    for path in sorted(REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        r = json.loads(path.read_text())
        items.append({"id": r["id"], "created_at": r["created_at"], "trial_id": r.get("trial_id"),
                      "verdict": r["aggregate"]["verdict"], "n_criteria": len(r["results"])})
    return items


def _report_path(report_id, ext):
    if not report_id.isalnum():
        raise HTTPException(400, "invalid report id")
    path = REPORTS / f"{report_id}.{ext}"
    if not path.exists():
        raise HTTPException(404, "report not found")
    return path


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    return json.loads(_report_path(report_id, "json").read_text())


@app.get("/api/reports/{report_id}/download")
def download_report(report_id: str, fmt: Literal["html", "json"] = "html"):
    path = _report_path(report_id, fmt)
    return FileResponse(path, filename=f"screening-report-{report_id}.{fmt}")


@app.post("/api/benchmark")
def benchmark(body: BenchmarkIn):
    require_ready()
    sents = split_sentences(body.note)
    return engine.benchmark(sents, body.criterion_type, body.criterion_text.strip(), body.modes,
                            body.max_new_tokens, body.stop_at_eos)


@app.post("/api/attention-check")
def attention_check(body: AttentionIn):
    return engine.attention_check(body.seq_len, body.causal, body.block)


@app.get("/api/modes")
def modes():
    return MODES


@app.get("/api/results")
def results():
    """Offline experiment outputs (training curves, metrics, GPU benchmarks) if present."""
    found = {}
    for path in sorted(RESULTS.rglob("*.json")):
        try:
            found[str(path.relative_to(RESULTS)).replace("\\", "/")] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return found


# --- frontend -----------------------------------------------------------------

if (FRONTEND / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    index = FRONTEND / "index.html"
    if path.startswith("api/"):
        raise HTTPException(404, "unknown API route")
    if not index.exists():
        return HTMLResponse("<h1>Frontend not built</h1><p>Run <code>npm run build</code> in "
                            "<code>frontend/</code>, or use Docker.</p>", status_code=503)
    candidate = FRONTEND / path
    if path and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(index)
