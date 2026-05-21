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
- **Leakage hides in process timing, not just in target columns.** Some features are populated at intake (safe), some during repair (only safe if the model is being scored mid-repair), some at close (post-hoc, unsafe for any forward-looking prediction). The pipeline introduces an explicit timing classification — INTAKE / TRAINING / POST_INTAKE / MEDIUM / POST_CLOSE — to keep that discipline mechanical rather than mental. The five-test automated audit catches the subtle violations that human review misses.
- **Predictions only matter if they translate to interventions.** A 4.60-day MAE is operationally meaningless without saying which 28 Market_Category × Channel segments to act on and which of four operational levers to pull. The downstream lever decomposition is the deliverable; the model is the input.

This project addresses all three within a reproducible single-machine pipeline.

The pipeline processes **2.19M repair records** across 4 source years, joins three operational tables (master records, parts ledger, callback records), engineers **41 leakage-audited features**, trains and compares **15 candidate models** (7 classifiers + 8 regressors), and produces a prioritized action plan for the worst-performing service segments.

## Headline results

Validated end-to-end on a strict 2026 out-of-time holdout:

| Metric | Value | Note |
|---|---|---|
| **Holdout AUC at T=5** | **0.809** | LightGBM, 41 features |
| **Holdout F1 at T=5** | **0.732** | At decision threshold 0.50 |
| **Holdout MAE** | **4.60 days** | 32% improvement over the 6.77-day mean baseline |
| Val → Holdout AUC gap | 0.011 | Strong out-of-time generalization |
| Holdout cohort size | 70,250 repairs | 2026 only — locked, used once |
| Training cohort | 1,060,649 repairs | 2023-2024 |
| Validation cohort | 509,930 repairs | 2025 |

These numbers are reproducible from this code on the source data. The lever decomposition shows the top operational priorities split across four levers — **46% engineer deployment, 25% parts logistics, 18% channel process, 11% repair complexity** — concentrated in 7 Market_Category × Channel segments that appear in the top 10 across every operating threshold (T=3, 5, 7, 10).

## Key Technical Decisions

The choices below are the ones that drove the result. Each came from a measured failure of the alternative or an explicit constraint in the data.

### 1. Strict temporal split with locked 2026 holdout

Train on 2023-2024, validate on 2025, hold out 2026 — used exactly once at the end. A random split would let the model see future engineer assignments, future parts logistics shifts, and future seasonal patterns — all of which leak in subtle ways the validation log loss wouldn't catch. The 0.011 AUC gap between validation (0.821) and holdout (0.809) is the strongest evidence the model generalizes across operating years; a wider gap would have signaled overfitting to 2025's specific operational rhythm.

### 2. Five-class timing taxonomy, not "use everything available"

Every feature gets one of five labels: INTAKE, TRAINING, POST_INTAKE, MEDIUM, POST_CLOSE. Only INTAKE and TRAINING-class features are deployment-safe for a model scored at repair intake. The MEDIUM-class features (`is_ter_repair`, `is_sealed_repair`, `eng_channel_risk`) are flagged for operational confirmation rather than silently included or excluded — the methodology surfaces the assumption rather than burying it. This taxonomy is enforced by the automated audit, so timing discipline doesn't depend on remembering it during feature engineering.

### 3. Two-track CORE / EXTENDED feature design, not one-size-fits-all

The 41 features split into two model-specific subsets. CORE (27 features) has 0% missingness and is safe for linear models after median fill. EXTENDED (37 features) adds DMS-dependent features for tree models that handle nulls natively — those features have ~75% missingness because only ~28% of repairs route parts through DMS. This isn't a stylistic choice: linear models on EXTENDED produce worse holdout performance because median fill on 75%-null features destroys signal; tree models on CORE leave money on the table. The split is informed by both validation metrics and operational deployability.

### 4. Five-test automated leakage audit, not human review

Human review catches obvious leakage (target columns, hash of identifier) and misses subtle leakage (target encoding without proper folding, future-data proxies, train-vs-holdout distribution shift). The audit runs five mechanical tests on every feature: correlation with target above class-conditional levels, perfect-predictor flag, target-encoding folding check, future-data proxy detection, train-vs-holdout stability. Of 41 features, 33 pass clean; 8 are flagged for human review with explicit reasons. The discipline is in catching what manual review wouldn't.

### 5. Comparative model evaluation, not asserted choice

Baseline → linear (Ridge, Lasso) → tree (depth-6) → ensemble (RF) → boosting (XGBoost, LightGBM), with the lift at each stage justifying the LightGBM production choice rather than asserting it. The progression from mean baseline MAE 6.77 days → LightGBM MAE 4.64 days is monotone at each model class. The 0.04-day gap between XGBoost and LightGBM is within noise; the choice was made on training efficiency (LightGBM ~25% faster) and native categorical handling.

### 6. Lever decomposition translates predictions to interventions

A model that predicts which repairs will run late is useless without saying what to do about it. The downstream prioritization layer scores every Market_Category × Channel segment (≥500 repairs, 28 segments total) against four operational levers — engineer deployment, parts logistics, channel process, repair complexity — using a deterministic scoring rule per lever. Each segment gets a primary and secondary lever assignment. This is the business deliverable; the model is the input.

### 7. NPS used only for post-hoc validation, never as a feature

NPS responses (Promoter / Passive / Detractor) were explicitly excluded from the training feature set. Including them would have been target leakage — NPS scores correlate strongly with repair duration. Using them post-hoc to validate the model's business relevance yields a 16.5-point promoter rate gap between the lowest-risk and highest-risk predicted buckets, confirming the model addresses a real customer-experience problem rather than just an internal operational metric.

## Repo structure

```
rtat-optimization/
├── src/
│   ├── ingestion.py              Step 1-4: Excel → integrated parquet
│   ├── eda.py                    Step 5: first-pass EDA + hypothesis list
│   ├── feature_engineering.py    Step 6: 41 model features + leakage review
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
| 6 | `feature_engineering.py` | Master parquets | `feature_train.parquet`, `feature_holdout.parquet` + 4 doc CSVs | <1 min |
| 7 | `modeling.py` | Feature parquets | 7 model pickles + 6 result CSVs | 2-3 min |
| 8 | `prioritization.py` | Models + features | 7 priority CSVs | <1 min |
| 9 | `leakage_audit.py` | Features + models | `leakage_audit.csv` | <1 min |

The dominant cost is Step 4B (parts ledger streaming) at 15-25 minutes — Excel I/O on ~3M rows. Everything after ingestion runs against parquets and completes in under 5 minutes combined.

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
python feature_engineering.py   # Step 6: 41 features, leakage review, data dictionary
python modeling.py              # Step 7: 7 classifiers + 8 regressors, threshold sweep
python prioritization.py        # Step 8: priority matrix, 4-lever decomposition
python leakage_audit.py         # Step 9: 5-test feature audit
```

Each file is independently re-runnable. Re-running any stage overwrites that stage's outputs with deterministic identical results given the same inputs.

## Model performance — full comparison

### Regression (target: continuous `target_days`)

| Model | Val MAE (days) | Val RMSE | Val R² |
|---|---|---|---|
| 1. Mean baseline | 6.77 | 10.98 | -0.001 |
| 2. Segment mean (tier × channel) | 5.97 | 10.19 | 0.138 |
| 3. Ridge (L2) | 5.10 | 9.06 | 0.320 |
| 4. Lasso (L1) | 5.08 | 9.05 | 0.320 |
| 5. Decision Tree (depth=6) | 4.87 | 8.93 | 0.338 |
| 6. Random Forest | 4.66 | 8.69 | 0.374 |
| 7. XGBoost | 4.65 | 8.64 | 0.381 |
| **8. LightGBM** | **4.64** | **8.64** | **0.381** |

Lasso zeroed 3 of 27 CORE features at α=0.01 — modest regularization. The progression from mean baseline (6.77d) to LightGBM (4.64d) is a clean monotone improvement at each model class. The 0.04d gap between XGBoost and LightGBM is within noise; the choice between them was made on training efficiency (LightGBM is ~25% faster) and native categorical handling.

### Classification at the primary threshold (OnTime_5)

| Model | Val AUC | Val F1 | Val Precision | Val Recall |
|---|---|---|---|---|
| 1. Majority class | — | 0.000 | 0.000 | 0.000 |
| 2. Logistic Regression (L2) | 0.791 | 0.711 | 0.675 | 0.751 |
| 3. Decision Tree (d=3, interpretable) | 0.769 | 0.713 | 0.638 | 0.806 |
| 4. Decision Tree (d=6) | 0.787 | 0.711 | 0.671 | 0.755 |
| 5. Random Forest | 0.814 | 0.735 | 0.671 | 0.812 |
| 6. XGBoost | 0.820 | 0.740 | 0.684 | 0.805 |
| **7. LightGBM** | **0.821** | **0.721** | **0.720** | **0.722** |

### Holdout (2026, locked out-of-time)

| Threshold | Val AUC | Holdout AUC | Holdout F1 | Notes |
|---|---|---|---|---|
| T=3 (strict) | 0.801 | 0.788 | 0.460 | 27.9% positive, `is_unbalance=True` |
| **T=5 (primary)** | **0.821** | **0.809** | **0.732** | 46.8% positive, balanced |
| T=7 (relaxed) | 0.850 | 0.830 | 0.840 | 62.3% positive |
| T=10 (loose) | 0.868 | 0.857 | 0.903 | 75.4% positive, `is_unbalance=True` |

Holdout regression MAE: **4.60 days** (RMSE 8.39, R² 0.358). The 0.011 AUC gap between validation (2025) and holdout (2026) is the strongest signal that the model is not overfit to validation-year idiosyncrasies.

## Feature audit summary

Of 41 features in the production model, the automated 5-test audit returns:

| Verdict | Count | Examples |
|---|---|---|
| ✓ CLEAN | 33 | All TRAINING and INTAKE-class features |
| ⚠ CONFIRM WITH OPS | 3 | `is_ter_repair`, `is_sealed_repair`, `eng_channel_risk` — MEDIUM timing class |
| ⚠ REVIEW | 2 | `parts_has_arrival_flag`, `parts_delivery_tier` — POST_CLOSE class |
| ⚠ STABILITY FLAG | 3 | `month_of_year`, `quarter`, plus one feature where 2026 holdout (partial year) shifts distribution from 2023-2025 training |

The three "STABILITY FLAG" features are flagged for human review, not for removal — the train-vs-holdout shift is by design (the holdout is a temporal slice covering Jan-Apr 2026, while training spans full calendar years). The two "REVIEW" features are documented as having uncertain assignment timing; the methodology explicitly uses `seg_delivery_days_hist` (a training-cohort segment median) as the deployment-safe substitute. See [`docs/methodology.md`](docs/methodology.md) for the full leakage discipline.

The strongest leakage check is Test 5 — engineer historical proxy stability across train and holdout. The validated gap is **0.74 days**, well within the 1.0-day tolerance, confirming the encoder was correctly scoped to training years only.

## Lever decomposition — operational output

The downstream business deliverable. Every Market_Category × Channel segment with ≥500 repairs (28 segments total) is scored against four operational levers and assigned a primary and secondary lever:

| Lever | Segments | Share | Scoring rule |
|---|---|---|---|
| Engineer deployment | 13 | 46% | Q4 engineer rate vs population (×1.30 strong, ×1.10 weak) |
| Parts logistics | 7 | 25% | Parts rate × 1.15 + delivery days × 1.30 |
| Channel process | 5 | 18% | `max(0, channel_risk_ordinal - 3)` |
| Repair complexity | 3 | 11% | Sealed rate × 1.30 + reclaim rate × 1.30 |

Top recommendation (3. Urban × DMS) and the full ranking are in `outputs/prioritization/final_recommendation.csv`.

## NPS post-hoc validation

NPS responses (Promoter / Passive / Detractor) were **explicitly excluded from features** to avoid target leakage. Using them only for post-hoc model validation: among 2025 NPS responders, predicted-risk-bucket strongly predicts NPS sentiment:

| Predicted risk | Promoter rate | Detractor rate |
|---|---|---|
| Very low (0-20%) | **73.5%** | 16.5% |
| Very high (80-100%) | **57.0%** | **29.0%** |
| **Gap** | **16.5pp** | **12.5pp** |

The model's predicted lateness drives a 16.5-point promoter gap — confirms that RTAT optimization addresses a real customer-experience problem, not just an operational metric.

## Methodology choices documented elsewhere

- [`docs/methodology.md`](docs/methodology.md) — full methodology writeup: cohort filter, target encoding with smoothing (k=30), engineer historical mean computation discipline, two-track CORE/EXTENDED feature design, hyperparameter selection, leakage audit philosophy
- [`docs/deck.md`](docs/deck.md) — stakeholder presentation (markdown source)
- [`data/README.md`](data/README.md) — full schema specification with channel glossary and cohort filter rules
- [`outputs/README.md`](outputs/README.md) — artifact catalog: every output file by stage with description

## Two-track feature design

The 41 features split into two model-specific subsets:

- **CORE (27 features)**: 0% missing, safe for linear models after median fill. Excludes DMS-dependent features (parts line count, order quantity, multi-line flag, etc.) which have ~75% missingness because only ~28% of repairs route parts through DMS.
- **EXTENDED (37 features in MODEL_FEATURES)**: adds DMS-dependent features for tree models that handle missing values natively. Excludes `parts_order_to_arrival_days_safe` (training-EDA only — the deployment-safe substitute is `seg_delivery_days_hist`).

This isn't a stylistic choice. Linear models on the EXTENDED set produce worse holdout performance because median fill on 75%-null features destroys signal; tree models on the CORE set leave money on the table. The design choice is informed by both validation metrics and operational deployability.

## What I would do differently

Honest reflections after building this:

- **Stronger time-series cross-validation.** The current setup splits 2023-2024 → train, 2025 → val, 2026 → holdout. A rolling expanding-window CV across all training quarters would give a tighter estimate of expected drift.
- **Operational confirmation of MEDIUM-timing features.** The three features flagged by the audit (`is_ter_repair`, `is_sealed_repair`, `eng_channel_risk`) currently rely on documented (but not operationally confirmed) intake-time availability. A 1-hour meeting with the ops team would either ratify the assumption or surface a real leakage.
- **Model monitoring plan.** This codebase trains and validates a model. A production deployment would need a drift detector on `engineer_hist_mean_rtat` distributions, weekly KS tests on incoming features, and a reclaim-rate monitor (since reclaims feed back into next month's training data).
- **Per-segment models.** The current model is monolithic. For the top 5-10 priority segments, a specialized per-segment model might outperform — at the cost of operational complexity.
- **Cost-weighted thresholds.** T=5 is the chosen operating point on classifier-balance grounds. A real deployment would weight false-positives (over-allocated resources) against false-negatives (missed delays causing NPS hits) with actual business costs.

## License

MIT — see [LICENSE](LICENSE).
