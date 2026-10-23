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


sh("nvidia-smi")
sh(f"git clone -q {REPO} repo && cd repo && git checkout -q {REF} && git rev-parse HEAD")
sh("pip install -q transformers==5.20.2 peft==0.21.2 && pip install -q --no-deps -e repo")
sh("python -c \"import torch, transformers, peft; "
   "print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)\"")
sh("cd repo && python scripts/train_lora.py --out /kaggle/working/lora_r16 --baseline")
sh("rm -rf repo")  # keep the downloadable output small
