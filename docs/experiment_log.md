# Experiment and decision log

Chronological record of design decisions, rejected alternatives, failures and fixes,
with the evidence behind each. Source material for the report's Methods, Results and
Limitations sections (rubric A3).

## Data

| Finding | Evidence | Decision |
|---|---|---|
| 1,015 annotations, 53 patients (all `sigir-*`), 103 trials | `scripts/prepare_data.py` | 37/8/8 split matches exactly |
| Dataset ships its own `training` flag (881/134) | dataset schema | Ignored; prescribed seed-42 split used instead; dropped at load time |
| `explanation_correctness` judges GPT-4 output | dataset schema | Treated as leakage, dropped with the `gpt4_*` columns |
| Strong class imbalance; test split has **0** `excluded` | `data/splits/split_manifest.json` | Macro-F1 over gold-present classes; zero-support classes reported, not scored |
| 610/1,015 records have no gold evidence | manifest | Evidence micro-PRF over all records + macro over non-empty gold + empty-gold agreement |
| Annotation 883 has empty `criterion_text` | crash in `--longest-first` sort | Excluded from processed files, recorded in manifest (`dropped_annotations`) |
| Max prompt+target = 651 tokens (Qwen tokenizer) | `data/processed/token_stats.json` | `max_length=768` → no truncation; training raises instead of truncating |
| No expert explanations in the dataset | schema | Model trained to output label + evidence IDs; explanations assembled from cited sentences only |

## Model choice

* Target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct: identical tokenizer (required for
  speculative decoding), Apache-2.0, fits T4 training with LoRA and laptop CPU inference.
* Alternatives considered: Llama-3.2-1B/3B-Instruct (Llama 3.2 Community License, gated download;
  1B/3B pair also shares a tokenizer), Qwen2.5-3B/7B (better quality expected but CPU inference in the
  app would be too slow on a 16 GB laptop; 7B does not fit full-precision LoRA on a T4).

## Training

| Attempt | Outcome | Cause | Fix |
|---|---|---|---|
| v1 | Crash at LoRA injection | Kaggle image ships torchao 0.10; peft 0.21 requires > 0.16 | `pip uninstall torchao` (unused) |
| v2 | CUDA OOM (1.29 GiB alloc) in training step | batch 4 x 651 tok x 151,936-vocab fp32 logits = 1.6 GB per copy | batch 2 x grad-accum 4 (same effective batch 8), gradient checkpointing |
| v3 | CUDA OOM (4.39 GiB) after epoch 1 | validation loss used generation batch 16 | val loss uses train batch size; adapters saved before evaluation; preflight mirrors real batch sizes on the longest examples |
| v4 | Success: 3 epochs, 809 s, peak 5.49 GB | — | — |

Preflight now runs a short training on the longest examples with the real batch sizes before the
main run, so environment and memory failures cost seconds of GPU time, not minutes.

## Decoding / evaluation

* Qwen's `generation_config.json` sets `repetition_penalty=1.1` (plus sampling defaults); HF
  `generate` applies the penalty even with `do_sample=False`. Our KV-cache decoder disagreed with HF
  greedy at token 3 until `repetition_penalty=1.0` was forced. All evaluations use pure greedy.
* `min_new_tokens` in HF `generate` suppresses EOS logits; for fixed-length comparisons HF is called
  with `eos_token_id=None` instead, after which outputs match ours token-for-token.

## Attention

* Custom attention backends registered with transformers' `AttentionInterface` receive
  `attention_mask=None` unless a mask builder is also registered under the same name
  (`AttentionMaskInterface`); padded batches then silently attended to pad tokens (max logit
  error 10.6). Registering `sdpa_mask` fixed it (max error 2.6e-5, identical tokens).

## Application

* Prefix KV-cache reuse across criteria of one patient: ~234 of ~360 prompt tokens reused per
  criterion. A cache-reset with an empty token list produced a negative cache length (fixed with a
  clamp; benchmark modes use fresh caches).
* CPU latency (i5-1135G7, fp32, fine-tuned 1.5B): ~12 s per criterion (first 17 s).
* **Rejected:** dynamic int8 quantization of all Linear layers (`torch.ao.quantization.quantize_dynamic`):
  0.55x speed (slower) on this CPU and 2/12 correct labels vs 10/12 for fp32 on 12 val prompts.

## End-to-end application checks (Docker, CPU, published adapter)

* App API vs the offline evaluation predictions (Kaggle T4) on 16 validation criteria of two
  patient/trial groups: 16/16 identical labels and 16/16 identical evidence sets.
* Unseen patient (written for the demo, not from the dataset), six criteria: identical outputs in
  KV-cached, tiled-attention and speculative modes; 5/6 clinically reasonable. The error is
  "HbA1c between 7.0% and 10.0%" -> not enough information while citing "HbA1c is 8.4%": the same
  numeric-threshold under-inference seen in the test error analysis.
* Downloaded HTML report contains the disclaimer, every criterion and the provenance.
* ClinicalTrials.gov returns Markdown (`* 1\. text`, escaped punctuation, criteria split into
  several un-bulleted paragraphs). The first parser kept "1\." prefixes and split one criterion
  into three; fixed (stacked prefixes stripped, un-bulleted lines continue the previous item,
  Markdown escapes removed) and covered by a test.
* Speculative mode in a 7.6 GiB Docker VM loaded the draft and the container was killed (restart
  count 1). Fixed with a memory check that returns a 503 explanation; with WSL memory=10GB the
  draft fits (peak 8.3 GiB).

## Results so far (validation, greedy)

| Model | Acc | Macro-F1 | Evidence micro-F1 |
|---|---|---|---|
| Qwen2.5-1.5B zero-shot | 0.209 | 0.142 | 0.149 |
| + LoRA r16, 3 epochs (best epoch 3) | 0.748 | 0.446 | 0.544 |

Zero-shot predicts `excluded` for 110/140 exclusion criteria (polarity confusion). After
fine-tuning, rare classes (`not included`, `excluded`, inclusion `not applicable`) have 0 recall,
motivating the oversampling ablation.

## Ablations (validation only; best epoch by val macro-F1)

| Variant | Best ep | Acc | Macro-F1 | Ev. micro-F1 | Train s (T4) |
|---|---|---|---|---|---|
| main: evidence-first, r16, 3 ep | 3 | 0.748 | 0.446 | 0.544 | 809 |
| + rare-label oversampling (sqrt, cap 8; 602 -> 1,060 ex/epoch) | 1 | 0.738 | 0.526 | 0.489 | 1,290 |
| label-first answer order | 3 | 0.743 | 0.545 | 0.551 | 822 |
| 5 epochs | 5 | 0.752 | 0.454 | 0.577 | 1,378 |
| draft Qwen2.5-0.5B, main recipe (zero-shot 0.063 / 0.022 / 0.0) | 2 | 0.733 | 0.430 | 0.448 | 457 |

* Per-class counts show label-first's +0.10 macro-F1 comes from **one** extra correct `excluded`
  (1/2 vs 0/2); val has 3 `not included`, 2 `excluded`, 1 inclusion `not applicable`.
* Oversampling shifts recall toward minority classes (included 23->32/35, exclusion NEI 7->13/26)
  at the cost of `not excluded` (95->82/101); val loss rises after epoch 1 (overfitting the repeats).
* Paired bootstrap over val annotations (B=2000, `results/ablations/bootstrap_vs_main.json`): no
  variant's accuracy, macro-F1 or evidence-F1 difference vs main has a 95% CI excluding 0;
  oversampling lowers evidence F1 in 95.2% of resamples.
* **Decision:** keep the pre-declared selection rule (best val macro-F1) -> label-first r16 3 ep,
  reporting that its advantage is not statistically reliable. The draft is retrained label-first so
  its outputs follow the same schema as the target (otherwise acceptance would collapse).
