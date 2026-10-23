"""Kaggle GPU job: zero-shot baseline + LoRA fine-tuning of Qwen2.5-1.5B-Instruct.

Runs non-interactively and exits when done, so the GPU session ends with it.

  kaggle kernels push -p kaggle/train_lora --accelerator NvidiaTeslaT4 -t 7200
  kaggle kernels output mustafafaraz/trialmatch-train-lora -p kaggle_runs/train_lora/output
"""

import subprocess

REPO = "https://github.com/fsdev87/generative-ai-A2"
REF = "main"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


try:
    sh("nvidia-smi")
    sh(f"git clone -q {REPO} repo && cd repo && git checkout -q {REF} && git rev-parse HEAD")
    # Kaggle ships torchao 0.10, which peft >= 0.18 rejects at LoRA injection time; we do not use it.
    sh("pip uninstall -q -y torchao")
    sh("pip install -q transformers==5.20.2 peft==0.21.2 && pip install -q --no-deps -e repo")
    sh("python -c \"import torch, transformers, peft; "
       "print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)\"")
    # Preflight on the 4 longest examples: environment errors and peak-memory OOMs
    # fail in seconds instead of after the baseline.
    env = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    # Same batch sizes as the real run so every memory peak (train step, val loss,
    # batched generation) is exercised.
    sh(f"cd repo && {env} python scripts/train_lora.py --out /tmp/preflight --epochs 1 "
       "--limit-train 8 --longest-first --limit-val 16")
    sh(f"cd repo && {env} python scripts/train_lora.py --out /kaggle/working/lora_r16 --baseline")
finally:
    sh("rm -rf repo")  # keep the downloadable output small
