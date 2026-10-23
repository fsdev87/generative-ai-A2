# Clinical Trial Criterion Matching with a Fine-Tuned Transformer

GenAI Assignment 2. Given a patient note and **one** inclusion/exclusion criterion of a clinical
trial, a LoRA-fine-tuned **Qwen2.5-1.5B-Instruct** predicts the expert eligibility label and cites
the supporting patient-sentence IDs. The repository also contains from-scratch implementations of
**KV-cached decoding**, **tiled online-softmax attention** and **greedy speculative decoding**, and a
Dockerized **React + FastAPI** application for screening and live inference benchmarking.

> **Disclaimer:** educational screening prototype, not a clinical enrollment decision tool.

## Quick start (application)

Requirements: Docker Desktop (8 GB RAM available to Docker recommended).

```bash
# Windows: double-click start.bat      macOS/Linux: ./start.sh
docker compose up -d --build           # equivalent manual command
```

Open <http://localhost:8000>. On first start the base model (~3 GB) and the LoRA adapter are
downloaded into a Docker volume and loaded in the background; the status badge (top right) turns
green when ready. Stop with `docker compose down`.

By default the app downloads the fine-tuned LoRA adapter from Hugging Face:
**[farazmust123/qwen2.5-1.5b-trialgpt-lora](https://huggingface.co/farazmust123/qwen2.5-1.5b-trialgpt-lora)**.
To run the untuned base model instead (before/after comparison), put `ADAPTER=` in a `.env` file
next to `docker-compose.yml`.

| Variable | Default | Meaning |
|---|---|---|
| `ADAPTER` | `farazmust123/qwen2.5-1.5b-trialgpt-lora` | LoRA adapter (HF repo id or path) merged into the target model |
| `TARGET_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | target model |
| `DRAFT_MODEL` / `DRAFT_ADAPTER` | `Qwen/Qwen2.5-0.5B-Instruct` / none | draft model for speculative decoding |
| `ANSWER_ORDER` | `label_first` | output schema the adapter was trained with |
| `DTYPE` | `float32` | model dtype on CPU |
| `MAX_NEW_TOKENS`, `SPEC_K` | `64`, `4` | generation budget, speculative draft length |

### Without Docker (development)

```bash
python -m venv .venv && .venv/Scripts/activate            # Windows (source .venv/bin/activate elsewhere)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
cd frontend && npm ci && npm run build && cd ..
ADAPTER=path/or/repo uvicorn app.backend.main:app --port 8000
```

## Application features

* **Screen patient** — paste a note (sentences are numbered automatically) or load a validation-split
  example; fetch a trial's criteria from ClinicalTrials.gov by NCT ID or paste them; results stream
  per criterion with label, cited sentence IDs, highlighted evidence, an explanation assembled only
  from cited sentences, and flags (*unsupported*, *missing information*, *unparseable*). The
  preliminary overall outcome is an explicit rule-based aggregation (see About page). Reports are
  saved automatically and downloadable as HTML/JSON with provenance (model, adapter, decoding).
* **Inference benchmark** — runs the same prompt live through uncached, KV-cached, KV-cached with our
  tiled attention, and speculative decoding; shows prefill/decode time, tokens/s, token equality and
  draft acceptance; plus a live attention-kernel accuracy/timing check.
* **Experiments** — training curves, before/after metrics, confusion matrices and GPU benchmarks
  rendered from the JSON files in `results/`.

## Repository layout

```
src/trialmatch/
  data.py          dataset loading (leaky columns dropped), prescribed 37/8/8 split
  prompting.py     prompt, JSON output schema, robust output parser
  metrics.py       accuracy, macro-F1, per-class PRF, confusion matrices, evidence PRF
  modeling.py      model loading, batched greedy generation (HF) for evaluation
  evaluation.py    predict + score
  attention.py     reference SDPA and tiled online-softmax attention (+ HF backends)
  qwen_decoder.py  hand-written Qwen2 forward pass with an explicit, preallocated KV cache
  decoding.py      greedy uncached / KV-cached (with prefix reuse) / speculative decoding
scripts/
  prepare_data.py  split manifest, processed JSONL, token statistics
  train_lora.py    LoRA fine-tuning (plain PyTorch loop) with per-epoch validation selection
  evaluate.py      evaluation on val, or on test with --final
  bench_attention.py, bench_decoding.py   Task 3-5 benchmarks
kaggle/            Kaggle GPU jobs (script kernels that clone this repo and exit when done)
app/backend/       FastAPI server (engine, text parsing, reports)
frontend/          React + Vite UI
tests/             attention, decoding, API tests (pytest)
data/              split manifest and processed splits
results/           metrics, training logs and benchmark logs
```

## Reproducing the experiments

```bash
python scripts/prepare_data.py                      # split + manifest + token stats
pytest tests -q                                     # attention / decoding / API correctness
python scripts/train_lora.py --out runs/lora_r16 --baseline        # GPU recommended
python scripts/evaluate.py --split val --adapter runs/lora_r16/best --out results/eval/lora_val
python scripts/evaluate.py --split test --final --adapter runs/lora_r16/best --out results/eval/lora_test
python scripts/bench_attention.py --out results/benchmarks/attention.json
python scripts/bench_decoding.py --adapter runs/lora_r16/best --out results/benchmarks/decoding.json
```

All GPU runs in the report were executed as Kaggle script kernels on one NVIDIA T4
(`kaggle/*/run.py`), launched with `kaggle kernels push -p kaggle/<job> --accelerator NvidiaTeslaT4`.

## Dataset and split

[ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations):
1,015 criterion-level annotations, 53 patients, 103 trials. Split: sort unique `patient_id`,
`numpy.random.default_rng(42).shuffle`, first 37 train / next 8 val / last 8 test; every annotation of
a patient is in exactly one split ([manifest](data/splits/split_manifest.json), checks run on every
build). `gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`, `explanation_correctness` and the
dataset's own `training` flag are dropped at load time and never used. Annotation 883 (empty criterion
text) is excluded from the processed files and recorded in the manifest. The test split is only read by
`evaluate.py --final`.

## AI assistance

This project was developed with the help of an AI coding assistant (Claude); see the AI-use appendix
of the report. All code was tested as described above.
