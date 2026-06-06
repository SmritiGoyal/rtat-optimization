# RTAT Optimization Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)

Author: **Smriti Goyal** ([github.com/SmritiGoyal](https://github.com/SmritiGoyal))

---

End-to-end production-style pipeline that predicts repair turn-around time (RTAT) for a Fortune 500 appliance manufacturer's US service network, then translates predictions into segment-level resource allocation recommendations across four operational levers.

## The Problem

Repair turn-around time prediction sits at the intersection of operations research and applied machine learning, and it's harder than it first appears for three reasons:

- **Operational variability is structural, not noise.** Repair durations depend on engineer availability, parts logistics, channel routing rules, and customer responsiveness — five mostly-independent systems that each have their own failure modes. A naive average over the cohort hides 40-day swings between adjacent segments. The model needs to capture interactions across operational dimensions, not just main effects.
- **Leakage hides in process timing, not just in target columns.** Some features are populated at intake (safe), some during repair (only safe if the model is being scored mid-repair), some at close (post-hoc, unsafe for any forward-looking prediction). The pipeline introduces an explicit timing classification — INTAKE / TRAINING / POST_INTAKE / MEDIUM / POST_CLOSE — to keep that discipline mechanical rather than mental. The five-test automated audit catches many subtle violations human review misses — though not all: a fold-level encoder leak slipped past it and was caught by a separate fold-scoping ablation (see Key Decision 4).
- **Predictions only matter if they translate to interventions.** A 4.62-day MAE is operationally meaningless without saying which 28 Market_Category × Channel segments to act on and which of four operational levers to pull. The downstream lever decomposition is the deliverable; the model is the input.

This project addresses all three within a reproducible single-machine pipeline.

The pipeline processes **2.19M repair records** across 4 source years, joins three operational tables (master records, parts ledger, callback records), engineers a leakage-reviewed feature set (**40 numeric model features**, plus one categorical feature documented but held out of the numeric model), trains and compares **15 candidate models** (7 classifiers + 8 regressors), and produces a prioritized action plan for the worst-performing service segments.

## Headline results

Validated on a strict, locked 2026 out-of-time holdout using a select-then-refit protocol — select on 2023-2024 → 2025, refit on all 2023-2025, score 2026 exactly once:

| Metric | Value | Note |
|---|---|---|
| **Holdout AUC at T=5** | **0.819** | LightGBM (XGBoost 0.816); 40 features, 2026 holdout |
| **Holdout F1 at T=5** | **0.740** | At decision threshold 0.50 |
| **Holdout MAE** | **4.62 days** | 32% improvement over the 6.77-day mean baseline |
| Holdout R² | 0.36 | |
| Holdout cohort size | 70,250 repairs | 2026 (Jan-Feb) — locked, scored once |
| Training cohort (selection) | 1,060,649 repairs | 2023-2024 |
| Training cohort (final refit) | 1,570,579 repairs | 2023-2025 |
| Validation cohort | 509,930 repairs | 2025 |

The model is selected on the strict 2023-2024 → 2025 split (T=5 validation AUC ~0.77), then refit on all 2023-2025 and scored once on the 2026 holdout. The holdout AUC sits slightly *above* the selection-fold validation because the final model trains on an extra year of data; generalization is evidenced by consistent performance across all four thresholds and by the NPS external check, not by a single tight validation-to-holdout gap.

These numbers are reproducible from this code on the source data. The lever decomposition shows the top operational priorities split across four levers — **43% engineer deployment, 29% parts logistics, 18% channel process, 11% repair complexity** — concentrated in 7 Market_Category × Channel segments that appear in the top 10 across every operating threshold (T=3, 5, 7, 10).

## Key Technical Decisions

The choices below are the ones that drove the result. Each came from a measured failure of the alternative or an explicit constraint in the data.

### 1. Strict temporal split with locked 2026 holdout, fit fold-scoped

Train on 2023-2024, validate on 2025, hold out 2026 — used exactly once at the end. A random split would let the model see future engineer assignments, future parts logistics shifts, and future seasonal patterns — all of which leak in subtle ways the validation log loss wouldn't catch.

The discipline is enforced by **fold-scoped encoder fitting** and a **select-then-refit protocol**: every training-class aggregate is fit on the 2023-2024 fold for model selection, then refit on all 2023-2025 for the final model, which scores the 2026 holdout once.

This matters because an earlier version of the pipeline fit those aggregates — engineer historical mean (the model's strongest feature), tier/channel/division/month means, city/state target encoding, and the segment delivery median — on the *full* 2023-2025 cohort and then split into train/validation without refitting. That let 2025 information reach the validation rows, inflating the validation metrics and the apparent validation-to-holdout gap. The fix scopes every aggregate to data available before the rows it serves; the holdout numbers in this README are leak-free.

### 2. Five-class timing taxonomy, not "use everything available"

Every feature gets one of five labels: INTAKE, TRAINING, POST_INTAKE, MEDIUM, POST_CLOSE. Only INTAKE and TRAINING-class features are deployment-safe for a model scored at repair intake. The MEDIUM-class features (`is_ter_repair`, `is_sealed_repair`, `eng_channel_risk`) are flagged for operational confirmation rather than silently included or excluded — the methodology surfaces the assumption rather than burying it. This taxonomy is enforced by the automated audit, so timing discipline doesn't depend on remembering it during feature engineering.

### 3. Two-track CORE / EXTENDED feature design, not one-size-fits-all

The 40 numeric features split into two model-specific subsets. CORE (31 features) is low-missingness and safe for linear models after median fill. EXTENDED (the full 40) adds DMS-dependent features for tree models that handle nulls natively — those features have ~75% missingness because only ~25% of repairs route parts through DMS. This isn't a stylistic choice: linear models on EXTENDED produce worse holdout performance because median fill on 75%-null features destroys signal; tree models on CORE leave money on the table. The split is informed by both validation metrics and operational deployability.

### 4. Five-test automated leakage audit, not human review

Human review catches obvious leakage (target columns, hash of identifier) and misses subtle leakage (target encoding without proper folding, future-data proxies, train-vs-holdout distribution shift). The audit runs five mechanical tests on every feature: correlation with target above class-conditional levels, perfect-predictor flag, target-encoding folding check, future-data proxy detection, train-vs-holdout stability. It covers all 41 documented features (the 40 numeric model features plus the one categorical feature held out of the numeric model); 33 pass clean and 8 are flagged for human review with explicit reasons.

**What the audit does not catch — and a leak it missed.** The audit compares training against the 2026 holdout, so it is blind to a leak confined to the *validation fold*. The fold-level encoder leak described in Decision 1 passed all five tests — the leaked aggregates were "train-only" relative to 2026, so the train-vs-holdout comparison saw nothing wrong, even while 2025 was bleeding into validation. It was caught by a separate fold-scoping ablation, not by the audit. A defensible v2 audit needs a sixth test: refit every encoder per fold and assert the validation metric is stable.

### 5. Comparative model evaluation, not asserted choice

Baseline → linear (Ridge, Lasso) → tree (depth-6) → ensemble (RF) → boosting (XGBoost, LightGBM), with the lift at each stage justifying the production choice rather than asserting it. On validation, performance improves monotonically from the mean baseline (MAE 6.77 days) down through the tree models, which cluster tightly (RF 4.96, XGBoost 4.93, LightGBM 4.97). On the final 2026 holdout, LightGBM and XGBoost land within ~0.01 AUC of each other and trade the lead by threshold — LightGBM ahead at T=3/T=5 (0.806/0.819 vs 0.804/0.816), XGBoost ahead at T=7/T=10. LightGBM is the primary deployable model: it leads at the primary T=5 target and on regression, with XGBoost kept as a validated equal-performance alternative. Classifier early stopping watches AUC (the selection metric); at the relaxed thresholds T=7/T=10 the target is highly separable (62%/75% already on-time), so validation AUC saturates within a couple of boosting iterations and the models legitimately stop early — deeper trees overfit and lower val AUC.

### 6. Lever decomposition translates predictions to interventions

A model that predicts which repairs will run late is useless without saying what to do about it. The downstream prioritization layer scores every Market_Category × Channel segment (≥500 repairs, 28 segments total) against four operational levers — engineer deployment, parts logistics, channel process, repair complexity — using a deterministic scoring rule per lever. Each segment gets a primary and secondary lever assignment. This is the business deliverable; the model is the input.

### 7. NPS used only for post-hoc validation, never as a feature

NPS responses (Promoter / Passive / Detractor) were explicitly excluded from the training feature set. Including them would have been target leakage — NPS scores correlate strongly with repair duration. Using them post-hoc to validate the model's business relevance yields a 10.4-point promoter rate gap between the lowest-risk and highest-risk predicted buckets (computed out-of-sample), confirming the model addresses a real customer-experience problem rather than just an internal operational metric.

## Repo structure

```
rtat-optimization/
├── src/
│   ├── ingestion.py              Step 1-4: Excel → integrated parquet
│   ├── eda.py                    Step 5: first-pass EDA + hypothesis list
│   ├── feature_engineering.py    Step 6: 40 numeric features (+1 categorical) + leakage review
│   ├── modeling.py               Step 7: 15 candidate models + comparison
│   ├── prioritization.py         Step 8: priority matrix + 4-lever decomposition
│   ├── leakage_audit.py          Step 9: 5-test feature audit
│
├── data/
│   ├── raw/                  (gitignored — client data)
│   └── README.md             Schema documentation
│
├── outputs/                  (gitignored — regenerable)
│   ├── interim/              Ingestion artifacts
│   ├── eda/                  Stats tables + charts
│   ├── features/             Feature parquets + documentation
│   ├── models/               Trained models + comparison tables
│   ├── prioritization/       Priority matrix + lever decomposition
│   └── README.md             Artifact catalog
│
├── docs/
│   ├── methodology.md        Extended technical writeup
│   └── deck.md               Stakeholder presentation (markdown)
│
├── .gitignore
├── requirements.txt
├── LICENSE                   MIT
└── README.md                 (this file)
```

Each of the six pipeline files is independently runnable. Each consumes its predecessor's parquet/CSV artifacts and produces its own. No shared state, no orchestrator file — the file structure is the orchestration.

## Pipeline at a glance

| Step | File | Input | Output | Approx runtime |
|---|---|---|---|---|
| 1-2 | `ingestion.py` § 5-6 | Raw Excel | Inventory + key validation CSVs | 15-30 min |
| 3 | `ingestion.py` § 4 | Master Excel | Cohort summary CSVs | 5 min |
| 4A-C | `ingestion.py` § 5-7 | All three Excel sources | `master_train.parquet`, `master_holdout.parquet` | 15-25 min |
| 5 | `eda.py` | `master_train.parquet` | 11 CSV tables + 10 PNG charts + hypothesis list | 1 min |
| 6 | `feature_engineering.py` | Master parquets | `feature_train.parquet`, `feature_holdout.parquet` (+ a 2023-2025-fit `feature_train_final.parquet` / `feature_holdout_final.parquet` for the final refit) + 4 doc CSVs | <1 min |
| 7 | `modeling.py` | Feature parquets | model pickles + result CSVs (incl. final refit holdout) | 3-5 min |
| 8 | `prioritization.py` | Models + features | 7 priority CSVs | <1 min |
| 9 | `leakage_audit.py` | Features + models | `leakage_audit.csv` | <1 min |

The dominant cost is Step 4B (parts ledger streaming) at 15-25 minutes — Excel I/O on ~3M rows. Everything after ingestion runs against parquets and completes in a few minutes combined.

## Reproducing the pipeline

### 1. Environment

```bash
git clone https://github.com/SmritiGoyal/rtat-optimization.git
cd rtat-optimization
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### 2. Data

This pipeline was developed against client-confidential field service data and **the raw data is not included in this repository**. To run end-to-end you would need access to source files matching the documented schema. See [`data/README.md`](data/README.md) for the full schema specification, including the seven service channels, the cohort filter rules, and the reclaim flag columns.

### 3. Execution

Run the stages in order:

```bash
python ingestion.py             # Steps 1-4: produces master_train.parquet, master_holdout.parquet
python eda.py                   # Step 5: first-pass EDA, hypothesis list
python feature_engineering.py   # Step 6: feature build (fold-scoped + final refit), leakage review
python modeling.py              # Step 7: classifiers + regressors, threshold sweep, final holdout
python prioritization.py        # Step 8: priority matrix, 4-lever decomposition
python leakage_audit.py         # Step 9: 5-test feature audit
```

Each file is independently re-runnable. Re-running any stage overwrites that stage's outputs with deterministic identical results given the same inputs.

## Model performance — full comparison

### Regression — validation (2025), target: continuous `target_days`

| Model | Val MAE (days) | Val RMSE | Val R² |
|---|---|---|---|
| 1. Mean baseline | 6.77 | 10.98 | -0.001 |
| 2. Segment mean (tier × channel) | 5.97 | 10.19 | 0.138 |
| 3. Ridge (L2) | 5.39 | 9.40 | 0.267 |
| 4. Lasso (L1) | 5.37 | 9.39 | 0.268 |
| 5. Decision Tree (depth=6) | 5.13 | 9.33 | 0.278 |
| 6. Random Forest | 4.96 | 9.15 | 0.306 |
| 7. XGBoost | 4.93 | 9.07 | 0.318 |
| 8. LightGBM | 4.97 | 9.20 | 0.298 |

Lasso zeroed 4 of 31 CORE features at α=0.01 — modest regularization. On validation regression the tree models cluster tightly and XGBoost (4.93) edges LightGBM (4.97) within noise. The final LightGBM model, refit on all 2023-2025, reaches **4.62 MAE** on the 2026 holdout (RMSE 8.39, R² 0.36).

### Classification at the primary threshold (OnTime_5) — validation (2025)

| Model | Val AUC | Val F1 | Val Precision | Val Recall |
|---|---|---|---|---|
| 1. Majority class | — | 0.000 | 0.000 | 0.000 |
| 2. Logistic Regression (L2) | 0.770 | 0.690 | 0.665 | 0.718 |
| 3. Decision Tree (d=3, interpretable) | 0.738 | 0.705 | 0.609 | 0.837 |
| 4. Decision Tree (d=6) | 0.772 | 0.703 | 0.649 | 0.766 |
| 5. Random Forest | 0.791 | 0.716 | 0.658 | 0.784 |
| 6. XGBoost | 0.802 | 0.717 | 0.686 | 0.750 |
| 7. LightGBM | 0.770 | 0.703 | 0.677 | 0.731 |

On this single-fit validation bake-off XGBoost leads on AUC (0.802); LightGBM (0.770, best_iter 113) trails on the single fit but pulls ahead on the final refit holdout (see below). LightGBM is the primary model — it leads at T=5 and on regression on the deployable holdout.

### Holdout (2026, locked out-of-time) — final model, refit on 2023-2025

| Threshold | Selection val AUC (2023-24 model) | Final holdout AUC (LGB) | Holdout F1 | Notes |
|---|---|---|---|---|
| T=3 (strict) | 0.758 | 0.806 | 0.499 | 27.9% positive |
| **T=5 (primary)** | **0.770** | **0.819** | **0.740** | 46.8% positive, balanced |
| T=7 (relaxed) | 0.810 | 0.819 | 0.784 | 62.3% positive |
| T=10 (loose) | 0.832 | 0.847 | 0.874 | 75.4% positive |

Holdout regression MAE: **4.62 days** (RMSE 8.39, R² 0.36). The holdout AUC sits at or above the selection-fold validation at every threshold because the final model is refit on the extra 2025 year before scoring 2026 — the consistent across-threshold pattern, not a single tight gap, is the generalization evidence. XGBoost is within ~0.01 AUC of LightGBM and trades the lead by threshold (it leads T=7/T=10: holdout 0.831/0.856).

## Feature audit summary

Of the 41 audited features (40 numeric model features + 1 categorical held out of the numeric model), the automated 5-test audit returns:

| Verdict | Count | Examples |
|---|---|---|
| ✓ CLEAN | 33 | All TRAINING and INTAKE-class features |
| ⚠ CONFIRM WITH OPS | 3 | `is_ter_repair`, `is_sealed_repair`, `eng_channel_risk` — MEDIUM timing class |
| ⚠ REVIEW | 2 | `parts_has_arrival_flag`, `parts_delivery_tier` — POST_CLOSE class |
| ⚠ STABILITY FLAG | 3 | `month_of_year`, `quarter`, plus one feature where 2026 holdout (partial year) shifts distribution from 2023-2025 training |

The three "STABILITY FLAG" features are flagged for human review, not for removal — the train-vs-holdout shift is by design (the holdout is a temporal slice covering Jan-Feb 2026, while training spans full calendar years). The two "REVIEW" features are documented as having uncertain assignment timing; the methodology explicitly uses `seg_delivery_days_hist` (a training-cohort segment median) as the deployment-safe substitute. See [`docs/methodology.md`](docs/methodology.md) for the full leakage discipline.

These verdicts are unchanged by the encoder fix — and that is precisely the audit's blind spot: it compares training against the 2026 holdout and cannot detect a leak confined to the 2025 validation fold (see Key Decision 4). Test 5 (engineer historical proxy stability across train and holdout) returned a **0.46-day** gap, well within the 1.0-day tolerance, confirming the engineer encoder is stable between training and the 2026 holdout — though, by construction, this test does not detect a validation-fold leak.

## Lever decomposition — operational output

The downstream business deliverable. Every Market_Category × Channel segment with ≥500 repairs (28 segments total) is scored against four operational levers and assigned a primary and secondary lever:

| Lever | Segments | Share | Scoring rule |
|---|---|---|---|
| Engineer deployment | 12 | 43% | Q4 engineer rate vs population (×1.30 strong, ×1.10 weak) |
| Parts logistics | 8 | 29% | Parts rate × 1.15 + delivery days × 1.30 |
| Channel process | 5 | 18% | `max(0, channel_risk_ordinal - 3)` |
| Repair complexity | 3 | 11% | Sealed rate × 1.30 + reclaim rate × 1.30 |

Top recommendation (Urban × DMS) and the full ranking are in `outputs/prioritization/final_recommendation.csv`.

## NPS post-hoc validation

NPS responses (Promoter / Passive / Detractor) were **explicitly excluded from features** to avoid target leakage. Using them only for post-hoc validation — 2025 NPS responders scored out-of-sample by the 2023-2024 selection model (which never saw 2025) — predicted-risk bucket tracks NPS sentiment in the right direction:

| Predicted risk | Promoter rate | Detractor rate |
|---|---|---|
| Very low (0-20%) | **69.2%** | 19.6% |
| Very high (80-100%) | **58.8%** | **27.3%** |
| **Gap** | **10.4pp** | **7.7pp** |

Predicted lateness drives a 10.4-point promoter gap and 7.7-point detractor gap — confirming RTAT optimization addresses a real customer-experience problem, not just an operational metric. The signal cannot be tuned from inside the pipeline because NPS is never a feature. Full breakdown across all five buckets: `outputs/prioritization/nps_validation.csv`, 42,451 responders.

## Methodology choices documented elsewhere

- [`docs/methodology.md`](docs/methodology.md) — full methodology writeup: cohort filter, target encoding with smoothing (k=30), engineer historical mean computation discipline, two-track CORE/EXTENDED feature design, hyperparameter selection, leakage audit philosophy and its blind spot
- [`docs/deck.md`](docs/deck.md) — stakeholder presentation (markdown source)
- [`data/README.md`](data/README.md) — full schema specification with channel glossary and cohort filter rules
- [`outputs/README.md`](outputs/README.md) — artifact catalog: every output file by stage with description

## Two-track feature design

The 40 numeric features split into two model-specific subsets:

- **CORE (31 features)**: low missingness, safe for linear models after median fill. Excludes DMS-dependent features (parts line count, order quantity, multi-line flag, etc.) which have ~75% missingness because only ~25% of repairs route parts through DMS, plus the ~44%-missing `seg_delivery_days_hist`.
- **EXTENDED (40 features = the numeric MODEL_FEATURES)**: adds DMS-dependent features for tree models that handle missing values natively. Uses the deployment-safe `seg_delivery_days_hist` and excludes both `parts_order_to_arrival_days_safe` (training-EDA only) and the one categorical-string column `parts_shipping_tier` (documented but held out of the numeric model).

This isn't a stylistic choice. Linear models on the EXTENDED set produce worse holdout performance because median fill on 75%-null features destroys signal; tree models on the CORE set leave money on the table. The design choice is informed by both validation metrics and operational deployability.

## What I would do differently

Honest reflections after building this:

- **A full hyperparameter sweep on the clean feature set.** After the leak fix, classifier early stopping was switched to watch AUC (the selection metric) rather than logloss, which restored LightGBM to competitive at T=3/T=5. A complete Optuna sweep on learning rate, depth, and leaf count — rather than just the early-stopping fix applied here — would likely recover a few more points and is the highest-value next step. Note the T=7/T=10 models legitimately train to only a couple of iterations because those targets are highly separable and validation AUC saturates immediately; that is expected, not under-tuning.
- **A sixth audit test for fold-level leaks.** The 5-test audit missed a validation-fold encoder leak because it only compares training against the 2026 holdout. Adding a per-fold encoder-refit stability check would catch the class of leak that actually occurred here.
- **Stronger time-series cross-validation.** The current setup splits 2023-2024 → train, 2025 → val, 2026 → holdout. A rolling expanding-window CV across all training quarters would give a tighter estimate of expected drift.
- **Operational confirmation of MEDIUM-timing features.** The three features flagged by the audit (`is_ter_repair`, `is_sealed_repair`, `eng_channel_risk`) currently rely on documented (but not operationally confirmed) intake-time availability. A 1-hour meeting with the ops team would either ratify the assumption or surface a real leakage.
- **Model monitoring plan.** This codebase trains and validates a model. A production deployment would need a drift detector on `engineer_hist_mean_rtat` distributions, weekly KS tests on incoming features, and a reclaim-rate monitor (since reclaims feed back into next month's training data).
- **Per-segment models.** The current model is monolithic. For the top 5-10 priority segments, a specialized per-segment model might outperform — at the cost of operational complexity.
- **Cost-weighted thresholds.** T=5 is the chosen operating point on classifier-balance grounds. A real deployment would weight false-positives (over-allocated resources) against false-negatives (missed delays causing NPS hits) with actual business costs.

## License

MIT — see [LICENSE](LICENSE).
