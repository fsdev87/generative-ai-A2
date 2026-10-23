# --- stage 1: build the React frontend -----------------------------------------
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- stage 2: FastAPI backend + model runtime (CPU) -----------------------------
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    REPORTS_DIR=/srv/reports
WORKDIR /srv

RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps -e .
COPY app/ app/
# Validation-split examples for the UI (the held-out test split is not shipped) and
# offline experiment results for the Experiments page.
COPY data/processed/val.jsonl data/processed/val.jsonl
COPY results/ results/
COPY --from=ui /ui/dist frontend/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
