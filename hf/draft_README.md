---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
datasets:
- ncbi/TrialGPT-Criterion-Annotations
tags:
- lora
- speculative-decoding
- clinical-trials
---

# Qwen2.5-0.5B-Instruct LoRA draft model for speculative decoding

Draft model paired with
[farazmust123/qwen2.5-1.5b-trialgpt-lora](https://huggingface.co/farazmust123/qwen2.5-1.5b-trialgpt-lora)
for greedy speculative decoding. Same tokenizer and answer format (label-first JSON) as the target;
trained with the same recipe (LoRA r=16, 3 epochs, best epoch by validation macro-F1) on the
training split of ncbi/TrialGPT-Criterion-Annotations. Fine-tuning raised draft-token acceptance on
screening prompts from 75.7% (base 0.5B) to 92.6% (k=4). Speculative decoding output is identical to
the target's greedy output by construction.

Code: https://github.com/fsdev87/generative-ai-A2 — **educational research prototype, not for
clinical use.**
