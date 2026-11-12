# Pipeline

Eight stages. Stage 0 is enough to check the paper's tables; stages 1–7 regenerate the
artefacts from scratch and need a GPU.

```
1  download models        02_download_model.py            ~40 GB
2  collect activations    03_collect_text_activations.py  ~20 min/model
3  train per-layer SAEs   35_train_per_layer_sae.py       1–3 h/model
4  single-feature effects 36_causal_validity.py           2–4 h/model
5  attribution baselines  47_attribution_baselines.py     ~1 h/combo
6  selectors & CFSE       40, 48, 49, 50, 77              ~10 h
7  statistics             43, 44, 94, 98                   CPU only
8  review follow-ups      95, 96, 97, 9A, 9H, 9I, 9J, 9M   ~6 h
```

All scripts live in `src/` and are run from the repository root, because they resolve
`config.json` and `output/` relative to the working directory.

## Stage 0 — tables only (no GPU, seconds)

```bash
pip install numpy
python scripts/make_tables.py
```

Writes `tables/*.csv` and `tables/*.md` from `results/`. Every number in those files is
byte-identical to the value the paper prints; `docs/NUMBERS.md` maps each table to the
result file it comes from.

## Stage 1 — models

```bash
python src/02_download_model.py                 # all models in config.json
python src/02_download_model.py --model pythia-1.4b
```

## Stage 2 — activations

```bash
python src/03_collect_text_activations.py --model pythia-1.4b
```

Writes one `.npz` per run; the newest file is picked up automatically downstream. The
layer suffix in the keys is what makes per-layer training possible.

## Stage 3 — per-layer SAEs

```bash
python src/35_train_per_layer_sae.py --model pythia-1.4b            # 3 layers, seed 42
python src/35_train_per_layer_sae.py --model pythia-1.4b --layers 16 --seed 43 \
       --out_dir output/phaseStep2/layer_sae_s43                    # seed-variance runs
```

`--num_epochs 2000` reproduces the paper (default is 120 for quick smoke tests).
The three seeds in `results/e10_summary.json` used 2000/2000/2000 epochs and reached
validation cosine 0.727 / 0.724 / 0.722, i.e. matched dictionary quality.

## Stage 4 — ground-truth single-feature effects

```bash
python src/36_causal_validity.py --model pythia-1.4b --layers 16 \
       --tasks ioi --n_prompts 200 --n_features 200
```

`--sae_root layer_sae_s43` points at an alternative dictionary; this is how the
seed-variance comparison (`9J`) is produced.

## Stage 5 — attribution baselines

```bash
python src/47_attribution_baselines.py --model pythia-1.4b --tasks ioi \
       --n_prompts 200 --shap_coalitions 128 --ig_steps 8
```

## Stage 6 — selectors and the held-out protocol

```bash
python src/40_selector_and_margin_benchmark.py --model pythia-1.4b
python src/49_cfse_heldout.py --models pythia-1.4b --tasks ioi --n_features 200
python src/77_grid_heldout.py --all
```

## Stage 7 — statistics (CPU)

```bash
python src/43_variance_decomposition.py
python src/94_e7_stats_recompute.py
python src/98_e7b_clustered_tests.py
```

## Stage 8 — review follow-ups

```bash
python src/95_e1_batch_noise.py --all          # batch/precision audit, table tab:repro
python src/96_e4_dose_response.py --all        # dose-response,        table tab:dose
python src/9A_ce_kl.py --all                   # CE/KL read-out,       table tab:cekl
python src/97_e3_sae_free.py --all             # SAE-free bases
python src/9H_c05_c06.py --all                 # KernelSHAP budget + standard IG
python src/9M_ig_path.py                       # both IG paths, same integrating code
python src/9I_c02_union_pool.py --all          # union-pool re-analysis
python src/9J_c14_seeds.py                     # SAE seed variance
python src/9K_e8_aggregate.py                  # -> output/phaseStep2/cfse_e8_shap_ig/e8_summary.json
python src/9L_e9_aggregate.py                  # -> output/phaseStep2/cfse_e9_unionpool/e9_summary.json
```

The aggregators are CPU-only and idempotent: re-running them after adding a combination
rewrites the summary file, and `scripts/make_tables.py` then regenerates `tables/`.

## Conventions that matter

- **Left padding.** Prompts in a batch are left-padded; `position_ids` are left implicit.
  Section 7.9 shows that forcing `position_ids = cumsum(mask) - 1` does not remove the
  batch-to-batch discrepancy, so the harness is not the cause. `9E_pad_diagnose.py`
  reproduces that check.
- **float32 read-outs.** Every effect is scored by recomputing logits from the lm_head
  weight in float32 (`9A`, `57_resolution_fix.py`), which is why bf16 storage does not
  change the rankings.
- **One protocol per comparison.** Numbers from the same-prompt and held-out protocols
  never enter the same statistic; each table states which one it uses.
