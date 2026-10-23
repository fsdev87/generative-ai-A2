"""Kaggle GPU job: draft LoRA (label-first), one-time final test evaluation, GPU benchmarks.

Final model (selected on validation macro-F1 before this job): LoRA r16, label-first answer
order, 3 epochs, from the trialmatch-ablations job, mounted read-only as an input.

  kaggle kernels push -p kaggle/final --accelerator NvidiaTeslaT4 -t 9000
  kaggle kernels output mustafafaraz/trialmatch-final -p kaggle_runs/final/output
"""

import glob
import subprocess

REPO = "https://github.com/fsdev87/generative-ai-A2"
REF = "main"
ENV = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
OUT = "/kaggle/working"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find_adapter(run_name):
    hits = glob.glob(f"/kaggle/input/**/{run_name}/best/adapter_config.json", recursive=True)
    if not hits:
        raise FileNotFoundError(f"adapter {run_name}/best not found under /kaggle/input")
    return hits[0].rsplit("/", 1)[0]


try:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"git clone -q {REPO} repo && cd repo && git checkout -q {REF} && git rev-parse HEAD")
    sh("pip uninstall -q -y torchao")
    sh("pip install -q transformers==5.20.2 peft==0.21.2 && pip install -q --no-deps -e repo")
    target = find_adapter("label_first")
    print("final adapter:", target, flush=True)

    # 1. Draft model fine-tuned with the same answer order as the target.
    sh(f"cd repo && {ENV} python scripts/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct "
       f"--answer-order label_first --out {OUT}/draft_label_first")
    sh(f"rm -rf {OUT}/draft_label_first/epoch_*")
    draft = f"{OUT}/draft_label_first/best"

    # 2. Before/after with the final prompt on identical examples; test is read only here.
    lf = "--answer-order label_first --batch-size 16"
    sh(f"cd repo && python scripts/evaluate.py --split val {lf} --out {OUT}/eval/zeroshot_val_label_first")
    sh(f"cd repo && python scripts/evaluate.py --split val {lf} --adapter {target} --out {OUT}/eval/final_val")
    sh(f"cd repo && python scripts/evaluate.py --split test --final {lf} --out {OUT}/eval/zeroshot_test")
    sh(f"cd repo && python scripts/evaluate.py --split test --final {lf} --adapter {target} --out {OUT}/eval/final_test")

    # 3. GPU benchmarks (Tasks 3-5).
    sh(f"cd repo && python scripts/bench_attention.py --out {OUT}/benchmarks/attention_gpu.json")
    sh(f"cd repo && python scripts/bench_decoding.py --adapter {target} --answer-order label_first "
       f"--out {OUT}/benchmarks/decoding_gpu_draft_base.json")
    sh(f"cd repo && python scripts/bench_decoding.py --adapter {target} --answer-order label_first "
       f"--draft-adapter {draft} --out {OUT}/benchmarks/decoding_gpu_draft_lora.json")
finally:
    sh("rm -rf repo")
