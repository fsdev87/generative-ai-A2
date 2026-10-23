"""Kaggle GPU job: seed variability of the final recipe (label-first, r16, 3 epochs).

The submitted model is the seed-42 run selected earlier; these runs only measure how much
the validation and test metrics vary with the random seed (LoRA init, shuffling, dropout).
No decision is based on them.

  kaggle kernels push -p kaggle/seeds --accelerator NvidiaTeslaT4 -t 7200
  kaggle kernels output mustafafaraz/trialmatch-seeds -p kaggle_runs/seeds/output
"""

import subprocess

REPO = "https://github.com/fsdev87/generative-ai-A2"
REF = "main"
ENV = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
OUT = "/kaggle/working"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


try:
    sh(f"git clone -q {REPO} repo && cd repo && git checkout -q {REF} && git rev-parse HEAD")
    sh("pip uninstall -q -y torchao")
    sh("pip install -q transformers==5.20.2 peft==0.21.2 && pip install -q --no-deps -e repo")
    for seed in (43, 44):
        run = f"{OUT}/label_first_seed{seed}"
        sh(f"cd repo && {ENV} python scripts/train_lora.py --answer-order label_first --seed {seed} --out {run}")
        sh(f"rm -rf {run}/epoch_*")
        sh(f"cd repo && python scripts/evaluate.py --split test --final --answer-order label_first "
           f"--batch-size 16 --adapter {run}/best --out {OUT}/eval/seed{seed}_test")
finally:
    sh("rm -rf repo")
