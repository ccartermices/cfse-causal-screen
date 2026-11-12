# Data and models

Nothing large is stored in this repository. This page says exactly what the pipeline
needs, how big it is, and how to get it.

## 1. Models

Ten open-weight models, downloaded through `src/02_download_model.py` (ModelScope is used
when `modelscope_id` is set in `config.json`; Hugging Face otherwise).

| key | HF id | hidden | probed layers | SAE width / k |
|---|---|---|---|---|
| `qwen3.5-0.8b` | Qwen/Qwen3.5-0.8B | 1024 | 8 / 16 / 22 | 8192 / 80 |
| `qwen3.5-2b` | Qwen/Qwen3.5-2B | 2048 | 8 / 16 / 22 | 16384 / 160 |
| `qwen2.5-7b-instruct` | Qwen/Qwen2.5-7B-Instruct | 3584 | 9 / 18 / 26 | 28672 / 280 |
| `minicpm5-1b` | OpenBMB/MiniCPM5-1B | 1536 | 8 / 16 / 22 | 12288 / 120 |
| `tinyllama-1.1b` | AI-ModelScope/TinyLlama-1.1B-Chat-v1.0 | 2048 | 7 / 14 / 21 | 16384 / 160 |
| `phi-3.5-mini` | microsoft/Phi-3.5-mini-instruct | 3072 | 10 / 21 / 30 | 24576 / 240 |
| `ministral-3-3b` | mistralai/Ministral-3-3B-Instruct-2512 | 3072 | 8 / 17 / 25 | 24576 / 240 |
| `pythia-1.4b` | EleutherAI/pythia-1.4b | 2048 | 8 / 16 / 22 | 16384 / 160 |
| `pythia-2.8b` | EleutherAI/pythia-2.8b | 2560 | 10 / 21 / 30 | 20480 / 200 |
| `pythia-6.9b` | EleutherAI/pythia-6.9b | 4096 | 10 / 21 / 30 | 32768 / 320 |

Total download: ~40 GB in bf16. `config.json` points at `./models/<org>/<name>`; change
`local_path` if you keep them elsewhere.

Models are loaded in bf16 exactly as published; two models ship fp8 weights on the Hub and
are dequantised in place at load time (`model_utils.model_needs_fp8`,
`dequantize_fp8_model`). The paper's numerical caveat comes from bf16, not from fp8: the
fp32 control in Section 7.9 reloads the same checkpoint in float32.

## 2. Activation corpus

`src/03_collect_text_activations.py` runs the model over a fixed text corpus and writes
`output/phase1/text_activations/<model>/text_activations_<ts>.npz`.

- 9,000 rows per layer (the `.npz` stores keys `<i>_<layer>`, one row per token position);
- the corpus is a mixture of natural text, code and templated prompts, cached under
  `data/`;
- per-layer rows are filtered by layer suffix before training, so a per-layer SAE never
  sees pooled activations from another depth (this is the correction described in
  `src/35_train_per_layer_sae.py`).

Total: ~120 MB per model.

## 3. Task prompts

No download is needed. `ioi`, `ioi_distract` and `gt` prompts are generated in code by
`TASK_BUILDERS` in `src/36_causal_validity.py`, with a fixed seed, so every run sees the
same 200 prompts per task. The number pool for `gt` is filtered per tokenizer.

## 4. What is *not* included

- Model weights, activation caches, SAE checkpoints (`*.pt`), and `output/` are git-ignored.
- The `results/` directory contains only the small JSON summaries that back the tables.
  Anything that would let you skip the pipeline (per-feature effect matrices, per-coalition
  SHAP values) is deliberately absent: those are what the pipeline is for.
