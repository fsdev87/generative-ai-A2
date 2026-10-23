# Clinical Trial Criterion Matching with a Fine-Tuned Transformer

Educational research prototype for GenAI Assignment 2. Given a patient note and a single inclusion/exclusion criterion of a clinical trial, a fine-tuned open-weight decoder (Qwen2.5-1.5B-Instruct + LoRA) predicts the expert eligibility label and cites supporting patient-sentence IDs.

The repository also contains from-scratch implementations of:

- KV-cached autoregressive greedy decoding (vs. an uncached reference)
- Tiled scaled dot-product attention with online softmax (vs. a reference implementation)
- Greedy speculative decoding with a Qwen2.5-0.5B-Instruct draft model

and a browser application (React frontend + FastAPI backend) for screening and inference benchmarking.

> **Disclaimer:** this is an educational screening prototype, not a clinical enrollment decision tool.

Setup, reproduction and startup instructions will be added as components land.

## Dataset

[ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations): 1,015 criterion-level annotations over 53 patients. Split: patient-grouped 37/8/8 with `numpy.random.default_rng(42)`. GPT-4 predictions, GPT-4 explanations, `explanation_correctness` and the dataset's own `training` flag are never used as inputs or for evaluation.
