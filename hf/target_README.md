---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
datasets:
- ncbi/TrialGPT-Criterion-Annotations
tags:
- lora
- clinical-trials
- eligibility
---

# Qwen2.5-1.5B-Instruct LoRA for clinical-trial criterion matching

LoRA adapter (r=16, alpha=32, all attention and MLP projections) that judges a single inclusion or
exclusion criterion against a patient note with numbered sentences and answers with JSON:

```json
{"label": "<label>", "evidence": [<sentence ids>]}
```

Labels: inclusion — `included`, `not included`, `not enough information`, `not applicable`;
exclusion — `excluded`, `not excluded`, `not enough information`, `not applicable`.

* Trained on the training split (37 patients, 602 criterion annotations) of
  [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)
  using the patient-grouped split with `numpy.random.default_rng(42)` (37/8/8 patients).
* 3 epochs, answer-only cross-entropy, label-first answer order; selected by validation macro-F1.
* Prompt format, training and evaluation code: https://github.com/fsdev87/generative-ai-A2
  (use `ANSWER_ORDER=label_first`).

**Educational research prototype — not for clinical enrollment decisions.**
