"""Kaggle GPU job: design-choice ablations (selected on validation only) + draft fine-tuning.

  oversample    r16, evidence-first, 3 epochs, rare-label oversampling
  label_first   r16, label-first answer order, 3 epochs
  epochs5       r16, evidence-first, 5 epochs
  draft_lora    Qwen2.5-0.5B-Instruct with the main recipe (draft model for speculative decoding)

  kaggle kernels push -p kaggle/ablations --accelerator NvidiaTeslaT4 -t 9000
  kaggle kernels output mustafafaraz/trialmatch-ablations -p kaggle_runs/ablations/output
"""

import subprocess

REPO = "https://github.com/fsdev87/generative-ai-A2"
REF = "main"
ENV = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
RUNS = {
    "oversample": "--oversample",
    "label_first": "--answer-order label_first",
    "epochs5": "--epochs 5",
    "draft_lora": "--model Qwen/Qwen2.5-0.5B-Instruct --baseline",
}


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


try:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"git clone -q {REPO} repo && cd repo && git checkout -q {REF} && git rev-parse HEAD")
    sh("pip uninstall -q -y torchao")  # incompatible with peft >= 0.18, unused
    sh("pip install -q transformers==5.20.2 peft==0.21.2 && pip install -q --no-deps -e repo")
    # Preflight with the new options on the longest examples (oversampling enlarges batches of rare labels).
    sh(f"cd repo && {ENV} python scripts/train_lora.py --out /tmp/preflight --epochs 1 --limit-train 8 "
       "--longest-first --limit-val 16 --oversample --answer-order label_first")
    for name, extra in RUNS.items():
        sh(f"cd repo && {ENV} python scripts/train_lora.py --out /kaggle/working/{name} {extra}")
        sh(f"rm -rf /kaggle/working/{name}/epoch_*")  # keep only best/ to limit output size
finally:
    sh("rm -rf repo")
