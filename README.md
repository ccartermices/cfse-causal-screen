# Causal validity in sparse autoencoder evaluation

Code and result artefacts for the paper *"Causal Validity in Sparse Autoencoder Evaluation:
A Ten-Model Audit and a Zero-Intervention Predictor"* (under review).

The paper asks a single interventional question of every SAE quality proxy in common use —
*if I ablate this feature, does the model's behaviour change?* — and answers it on ten
open-weight models, three depths each, 60 (model, layer, task) combinations, with per-layer
BatchTopK SAEs and measured single-feature ablation effects as ground truth.

Two findings drive the repository:

1. **The gap.** Across 10,200 feature–combination entries the best of six standard metrics
   (activation frequency, mean activation, max activation, reconstruction gain, context
   coherence, corpus frequency) reaches a pooled Spearman of +0.206 with the measured
   effect; a real LLM auto-interpretation judge reaches −0.004.
2. **The screen.** A first-order predictor built from one forward and one backward pass
   reaches +0.725 without intervening, and a three-level admission gate — calibrated on a
   20-point sweep plus a 48-combination grid, and agreeing with measurement on 47/48 of
   them — decides when the comparison is meaningful at all.

## Quick start (no GPU)

```bash
pip install numpy
python scripts/make_tables.py
```

This rebuilds every table in `tables/` from the JSON summaries in `results/`, which are the
exact files the paper's numbers were read from. Nothing is re-computed, so the output is a
direct check of the manuscript against the released artefacts.

The full pipeline — models, activations, SAEs, interventions, attribution baselines — is
documented in [`docs/PIPELINE.md`](docs/PIPELINE.md) and driven by
`scripts/run_pipeline.sh` (or `.ps1`). It needs a GPU and roughly 40 GB of model weights.

## Layout

```
config.json              model list, layers, SAE widths, output paths
src/                     pipeline; numeric prefixes are stage order
  model_utils.py         model resolution, fp8 handling, path templates
  01–05                  download, activations, SAE training
  27, 36, 40, 47         interventions, effects, selectors, attribution baselines
  43, 44, 94, 98         statistics
  48–50, 77–78           the CFSE protocol and the 48-combination grid
  9A–9L                  review follow-ups (batch noise, dose, CE/KL, SHAP budget,
                         union pool, SAE seeds), each with its own docstring
scripts/
  make_tables.py         results/ -> tables/ (CPU, seconds)
  run_pipeline.sh|ps1    end-to-end run for one model
results/                 the JSON summaries behind every reported number
tables/                  generated; not hand-edited
docs/                    DATA.md, PIPELINE.md, NUMBERS.md
```

## What the numbers rest on

Three habits are worth knowing before reading the code.

**Effects are protocol-relative, and the paper says so.** A single (feature, prompt) cell
has a median effect of 0.0195 logits against a bf16 rounding step of the residual stream;
the audit in §7.9 shows those cells are not reproducible across batch composition
(cell-level *r* = 0.64, re-measurement noise 1.3× the median effect) while the per-feature
aggregate, which is what the paper ranks, is (feature-level ρ = 0.78 across batch sizes,
0.965 between bf16 and float32). Under float32 weights the same matrix is reproducible to
3×10⁻⁶ logits. Read magnitudes as protocol-relative; read ranks as the result.

**One unit of inference per comparison.** The inference unit is the (model, layer, task)
combination, not the (combination, split) pair; the 960 pseudo-replicated entries in an
earlier draft are 48 combinations, and every test in §5 and §7 is computed that way.

**Numbers are generated, not typed.** Every value added in the revision flows from
`output/phaseStep2/*.json` through a renderer into the manuscript, and `docs/NUMBERS.md`
maps each table to the JSON file it came from. If a number and its file disagree, the file
is right.

## Reproducing the tables

| table | script | result file |
|---|---|---|
| Standard metrics vs effect | `40_selector_and_margin_benchmark.py` | `results/cfse_summary.json` |
| Admission gate | `94_e7_stats_recompute.py` | `results/e7_stats.json` |
| Batch / precision audit | `95_e1_batch_noise.py` | `results/e1_summary.json` |
| Dose–response | `96_e4_dose_response.py` | `results/e4_summary.json` |
| CE / KL read-out | `9A_ce_kl.py` | `results/e5_summary.json` |
| SAE-free bases | `97_e3_sae_free.py` | `results/e3_summary.json` |
| SHAP budget / standard IG | `9H_c05_c06.py` | `results/e8_summary.json` |
| IG path comparison (mean vs local baseline) | `9M_ig_path.py` | `results/e8_ig_path.json` |
| Union pool | `9I_c02_union_pool.py` | `results/e9_summary.json` |
| SAE seed variance | `9J_c14_seeds.py` | `results/e10_summary.json` |

## Four robustness checks

The revision adds four measurements that the first version argued rather than tested. All four
are in `results/` and reproduce as tables in `tables/`.

- **The pool was not doing the work.** Rebuilding the feature pool as a union on which no
  metric had a selection advantage moves nothing: predictor +0.722 against +0.713 on the
  M2-selected pool, IG +0.670 against +0.634, mean activation +0.240 against +0.201
  (`tab_union_pool`). No standard metric crosses 0.30 on either pool.
- **KernelSHAP at 128 coalitions is a floor, not a measurement.** Against exactly enumerated
  Shapley values on an eight-feature case, budgets of ~63/108/182 coalitions recover rank
  correlations of 0.917/0.952/0.964 with machine-precision efficiency; at 16 features,
  128-vs-512 agreement is 0.44 on the weak combination and 0.99 on the strong one.
- **Standard IG is weaker than the local variant the paper reports.** With a mean-activation
  baseline, completeness holds by construction (0.80 at 8 steps, 0.96 at 128) but the rank
  correlation falls from +0.708 to +0.288 on the measurable combination. Both are reported,
  the table row is labelled *local*, and the projection residual (0.95) explains the gap.
- **The dictionary is not the result.** Three SAEs of matched quality (validation cosine
  0.722–0.727) that share almost no feature indices (top-50 Jaccard 0.15–0.27) give
  ρ(M2) = +0.259 ± 0.041, ρ(M1) = −0.123 ± 0.075, ρ(M3) = +0.259 ± 0.063; no seed lets a
  standard metric reach 0.30 (`tab_sae_seeds`). Ranking quality transfers, feature identity
  does not.

## Environment

Python 3.10, PyTorch 2.11 (CUDA 12.8), transformers 5.16. Exact pins in
`requirements.txt`. Table regeneration needs only numpy; everything else needs a GPU with
≥ 24 GB for the 7B models (smaller models run comfortably on 8 GB).

## What is not here

Model weights, activation caches, SAE checkpoints and per-feature effect matrices are not
committed (`.gitignore`). The pipeline regenerates them; see `docs/DATA.md` for sizes and
download instructions.

## License

MIT for the code. The result artefacts in `results/` are released under CC BY 4.0.
