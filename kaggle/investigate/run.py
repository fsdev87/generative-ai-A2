"""Kaggle GPU job: cached-vs-uncached fp16 mismatch investigation and draft/target cost ratio.

  kaggle kernels push -p kaggle/investigate --accelerator NvidiaTeslaT4 -t 3600
  kaggle kernels output mustafafaraz/trialmatch-investigate -p kaggle_runs/investigate/output
"""

import glob
import subprocess

REPO = "https://github.com/fsdev87/generative-ai-A2"
REF = "main"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}/adapter_config.json", recursive=True)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0].rsplit("/", 1)[0]


try:
    sh(f"git clone -q {REPO} repo && cd repo && git checkout -q {REF} && git rev-parse HEAD")
    sh("pip uninstall -q -y torchao")
    sh("pip install -q transformers==5.20.2 peft==0.21.2 && pip install -q --no-deps -e repo")
    target, draft = find("label_first/best"), find("draft_label_first/best")
    sh(f"cd repo && python scripts/investigate_decoding.py --adapter {target} --draft-adapter {draft} "
       "--answer-order label_first --out /kaggle/working/decoding_investigation.json")
finally:
    sh("rm -rf repo")
