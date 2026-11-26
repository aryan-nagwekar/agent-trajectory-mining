# Doomed in the First Steps?

This project studies whether observable early actions in SWE-bench Verified logs predict run failure and transfer across coding-agent frameworks.

## Reproduce the final study

Run `make all` from this directory to build the analysis, figures, tables, slide outline, and NeurIPS-style PDF. Requirements are Python 3.11, a C++17 compiler, and Tectonic or a TeX distribution with `latexmk`. The command creates the Python environment when needed, downloads missing public traces, and reuses verified stage outputs.

Use `make quick` for an isolated smoke run on about 500 raw traces. Use `make demo` to launch the saved-model replay on localhost.

## Setup and pilot

```sh
make setup
make data
make inspect
make test
make normalize
```

Use Python 3.11. Set `PYTHON=/path/to/python3.11` if needed. Setup records the resolved dependency versions in `results/env.txt`. On macOS, LightGBM requires an OpenMP runtime when modeling begins. The Apple Silicon environment passed its import check using the library bundled with the pinned PyTorch wheel. [LightGBM installation guide](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html)

The [Phase 1 inspection](results/phase1_inspection.txt) shows native records and label joins. The pilot Parquet files are in `data/processed/pilot/`. Read the [pilot checks](results/pilot/normalization_checks.csv) for the per-framework label balance.

## Expanded normalization

```sh
make data CONFIG=configs/expanded.yaml LIMIT=0
make normalize CONFIG=configs/expanded.yaml
```

The full-cohort inputs are cached under `data/raw/`. The pipeline writes `runs.parquet`, `steps.parquet`, and `collapsed_sequences.parquet` in `data/processed/`. The first two use the proposal schemas. The collapsed file keeps a singleton token unchanged and appends `×2` or `×3+` to runs of identical tokens. Raw step indices are zero-based.

[Per-submission checks](results/normalization_checks.csv) compare the normalized failure rate to one minus the official resolve rate. [Exclusions](results/normalization_skips.csv) list unusable or officially ineligible instances. [Dataset summary](results/data_summary.csv) groups runs and steps by framework and outcome, and [other_bash diagnostics](results/other_bash_top30.csv) list the most frequent unmatched commands.

The seeded validation file is `data/interim/validation_sample.csv`. Fill the `human_action` and `human_outcome` columns, then run `.venv/bin/python scripts/score_validation.py` to measure agreement.

[Data sources and labels](DATA_CARD.md) · [Deviations](DEVIATIONS.md)

## Task splits and sequential patterns

Run `make splits` for the audit and Phase 3 split manifests. Run `make phase4-report` to regenerate the Phase 4 analyses and the [checkpoint report](report/phase4_report.md). Use `make test` for the parser, split, leakage, outcome, and pattern checks. The analysis targets use `configs/expanded.yaml` by default and reuse normalized Parquet files.

The validation scorer reads the hand-labeled CSV. Normalization preserves that file once it exists.

## Failure archetypes and prefix prediction

Run `make cluster`, `make predict`, and `make phase6-report` after Phase 3. The resulting [Phase 5–6 report](report/phase6_report.md) includes the LSH benchmark, per-checkpoint model metrics, and LightGBM ablations. Phase 6 saves one model bundle per fold and checkpoint under `results/models/` along with held-out and calibration scores in `data/processed/prefix_predictions.parquet`.

## Fixed-cohort, stopping policy, and transfer

Run `make archetype-sweep window-benchmark archetype-refit cohorts calibrate transfer phase8-report` in order after Phase 6. This regenerates the [Phase 8 checkpoint](report/phase8_report.md), including the fixed-cohort curve, paired archetype ablation, first-stop replay results, and task-disjoint framework transfer. Window benchmarking requires a C++17 compiler.

## Results

<!-- results:start -->

| Result | Value |
| --- | --- |
| Fixed-cohort LightGBM AUROC at t=5 / 20 / 40 | 0.653 ± 0.021 / 0.675 ± 0.017 / 0.723 ± 0.033 |
| Same-task pair AUROC at t=20 | 0.709 |
| Multi-checkpoint policy, α=0.10 | 9.83% character cost saved, 16.69% failures caught, 4.90% false stops [4.16%, 5.76%] |
| Held-framework AUROC at t=20 | OpenHands 0.577, SWE-agent 0.622, mini-SWE-agent 0.535 |
| Transfer failure | OpenHands 24.38% false stops at t=20 and α=0.10. Zero retained pattern overlap |
| Highest fixed-cohort AUROC at t=10 / 20 | LightGBM 0.671 ± 0.026 / Bag LR 0.678 ± 0.024 |
| Multi-checkpoint policy, α=0.05 | 5.02% character cost saved and 2.25% false stops |
| Highest-savings singles at α=0.05 / 0.10 | step 5 (7.94% saved, 5.03% eligible false stops) / step 5 (17.66% saved, 10.03% eligible false stops) |
| Best singles within the observed eligible budget at α=0.05 / 0.10 | step 15 (6.12% saved, 4.90% eligible false stops) / step 10 (10.98% saved, 9.96% eligible false stops) |

Realized multi-checkpoint false stops stayed below α. The measured savings are a conservative floor for feasible threshold relaxation on this replay cohort. Pattern indicators and archetype similarity show no consistent incremental prediction gain. The signal persists within difficulty bins and within same-task pairs.

At 23,456 windows, LSH checks 7,640,309 of 275,080,240 possible pairs at 98.57% recall. Candidate versus exact-pair slopes are 1.97 and 2.00. Runtime slopes are 1.56 and 1.57. Compiled exact search takes 7.26 s versus 7.65 s for LSH. The intended asymptotic benefit depends on sparse candidate growth.

[Numeric source](results/metrics.json) · [Phase 8 tables](report/phase8_report.md)

<!-- results:end -->

### Run the bundled replay

The repository includes a compact snapshot of normalized replay tokens, held-out scores, and calibrated thresholds. The demo uses this snapshot when the full local analysis outputs are unavailable. The raw trajectory cache and human-labeling workspace stay local.

From the project root, run `make setup` on a fresh checkout and then `make demo`. The Streamlit entry point for a hosted instance is `streamlit_app.py` with Python 3.11.
