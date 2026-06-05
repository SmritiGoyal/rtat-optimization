# Output Artifacts

This directory holds every artifact the pipeline produces, organized by stage. All content here is **regenerable** from the pipeline + the source data — none of it is committed to the repository.

## Folder structure

```
outputs/
├── interim/         Step 1-4 ingestion artifacts (master integration)
├── eda/             Step 5 first-pass EDA stats + charts
├── features/        Step 6 model-ready feature matrices + documentation
├── models/          Step 7 trained models + comparison tables + audit
└── prioritization/  Step 8 segment priority matrix + lever decomposition
```

Run any stage independently — each consumes its predecessor's artifacts.

---

## `outputs/interim/` — Ingestion

Produced by `ingestion.py` (Steps 1-4). Contains diagnostic CSVs and the integrated master parquet that all downstream stages consume.

| File | Source step | Description |
|---|---|---|
| `sheet_inventory.csv` | Step 1 | All workbook sheets with row/column counts, domain classification |
| `column_profile.csv` | Step 1 | Per-column null rate, cardinality, type guess, sample-based profile |
| `truncation_flags.csv` | Step 1 | Sheets that hit Excel's 1,048,576-row limit (typically 2024/2025 parts) |
| `domain_key_stats.csv` | Step 2 | Per (domain, year, key column): row counts, null rate, uniqueness |
| `pairwise_overlap.csv` | Step 2 | Master↔Parts and Master↔Reclaim key overlap per year + ALL |
| `parts_internal_compare.csv` | Step 2 | `Repair_Receipt_No` vs `Repair_Receipt_No_Merge` equality check |
| `cohort_summary.csv` | Step 3 | Per-year cohort sizes, exclusion counts, NPS coverage |
| `target_dist.csv` | Step 3 | Per-year target_days mean / median / p75 / p90 / p95 |
| `ontime_rates.csv` | Step 3 | Per-year on-time rates for T=1..10 |
| `reclaim_features.parquet` | Step 4A | Repair-level reclaim features (~2.08M rows, 26 cols) |
| `parts_features.parquet` | Step 4B | Repair-level parts features (~705K rows, 23 cols) |
| `parts_features_modelsafe.parquet` | Step 4B | Subset of parts features that are deployment-safe |
| `parts_*.parquet` | Step 4B | Per-sheet intermediate parts files (concatenated into the above) |
| `master_integrated.parquet` | Step 4C | **Full universe**: master joined with reclaim + parts (~2.19M rows) |
| `master_train.parquet` | Step 4C | Training cohort (2023-2025, ~1.57M rows; modeling.py splits into train+val) |
| `master_holdout.parquet` | Step 4C | Holdout cohort (2026 only, 70,250 rows — locked) |

---

## `outputs/eda/` — Exploratory Analysis

Produced by `eda.py` (Step 5). Stats tables for each EDA section plus matching PNG charts.

| File | Section | Description |
|---|---|---|
| `5a_target_distribution_by_year.csv` | 5A | Overall RTAT distribution: per-year mean / median / p75-p95 |
| `5a_overall_distribution.png` | 5A | 3-panel chart: histogram, year-over-year, OnTime_T rates |
| `5b_market_category_stats.csv` | 5B | Per-tier: volume, mean RTAT, on-time rates at T=3/5/7/10 |
| `5b_market_category.png` | 5B | Tier mean / on-time / priority-matrix scatter |
| `5c_channel_stats.csv` | 5C | Per-channel (top 8): mean RTAT, on-time rates, volume |
| `5c_channel_analysis.png` | 5C | Channel mean RTAT + on-time trend |
| `5d_division_stats.csv` | 5D | Per-division (top 10): mean RTAT, parts rate, on-time |
| `5d_division_analysis.png` | 5D | Above-average highlighting + parts-rate bubble chart |
| `5e_delivery_impact.csv` | 5E | RTAT by parts delivery bin (0-1d to 10d+) |
| `5e_parts_impact.png` | 5E | Parts overlay + complexity + delivery-bin trend |
| `5f_engineer_quartiles.csv` | 5F | Per-quartile engineer historical mean stats |
| `5f_engineer_signal.png` | 5F | Quartile bars + proxy-vs-actual scatter |
| `5g_monthly_stats.csv` | 5G | Per-month mean RTAT + on-time rate |
| `5g_time_effects.png` | 5G | Monthly seasonality + year-over-year |
| `5h_reclaim_signal.csv` | 5H | First-visit vs reclaim mean RTAT and on-time |
| `5i_nps_validation.csv` | 5I | NPS by RTAT bucket (2025 responders only) |
| `5i_nps_validation.png` | 5I | NPS by lead-time bucket + by Market_Category |
| `5j_segment_priority_matrix.csv` | 5J | Market_Category × Channel raw late rate matrix |
| `5j_priority_heatmap.png` | 5J | Color-coded late-rate heatmap (the key EDA visual) |
| `5k_hypothesis_list.csv` | 5K | The 8 hypotheses (H1-H8) used downstream |

---

## `outputs/features/` — Feature Engineering

Produced by `feature_engineering.py` (Step 6). The model-ready feature matrices plus four documentation CSVs.

| File | Description |
|---|---|
| `feature_train.parquet` | Train + val rows with all MODEL_FEATURES columns + targets + meta (~1.57M rows) |
| `feature_holdout.parquet` | Holdout rows with same columns (70,250 rows) |
| `feature_train_final.parquet` | 2023-2025 rows, aggregates fit on all 2023-2025 (Phase-2 final refit) |
| `feature_holdout_final.parquet` | 2026 holdout, aggregates fit on all 2023-2025 |
| `feature_spec.csv` | One row per feature: group, in_train, pct_missing, dtype, in_model |
| `data_dictionary.csv` | Full data dictionary: source, type, model_use, deployment_safe, description |
| `leakage_review.csv` | Per-feature: risk class (LOW/MEDIUM/HIGH) + rationale |
| `missingness_summary.csv` | Per-feature: group, pct_missing, dtype, unique count |

`MODEL_FEATURES` contains 41 entries across 8 groups — **40 numeric features** that feed the trained models, plus one categorical column (`parts_shipping_tier`) that is documented and leakage-audited but held out of the numeric model:

| Group | Count | Group | Count |
|---|---|---|---|
| Geography | 5 | Reclaim | 6 |
| Channel | 3 | Parts logistics | 11 |
| Time | 6 | Interactions | 3 |
| Product | 4 | **Total** | **41** |
| Engineer | 3 | | (40 numeric + 1 categorical) |

(`parts_order_to_arrival_days_safe` is in the feature parquet for EDA/audit reference but excluded from MODEL_FEATURES entirely.)

---

## `outputs/models/` — Modeling

Produced by `modeling.py` (Step 7) and updated by `leakage_audit.py` (Step 9).

### Result CSVs

| File | Description |
|---|---|
| `classification_results.csv` | All 7 classifiers at T=5: AUC, AP, P/R/F1, confusion matrix counts |
| `regression_results.csv` | All 8 regressors: MAE / RMSE / R² on validation |
| `threshold_results.csv` | LightGBM at T=3/5/7/10: AUC, P/R/F1, positive rate |
| `threshold_results_xgb.csv` | XGBoost per-threshold AUC/F1 |
| `threshold_sensitivity.csv` | Operating-point sweep: P/R/F1 at decision thresholds 0.10-0.90 |
| `feature_importance.csv` | Side-by-side LightGBM vs XGBoost importance with rank difference |
| `lasso_features.csv` | Lasso coefficients per CORE feature (4 zeroed of 31 at α=0.01) |
| `segment_performance.csv` | Per Market_Category: AUC of LightGBM and XGBoost |
| `channel_performance.csv` | Per Channel: AUC of LightGBM and XGBoost |
| `leakage_audit.csv` | Per-feature verdicts from the 5-test audit (41 audited, 33 CLEAN) |

### Model artifacts (joblib pickles)

| File | Description |
|---|---|
| `lgbm_ontime3.pkl` | LightGBM classifier for OnTime_3 (27.9% positive) |
| `lgbm_ontime5.pkl` | **Primary classifier** for OnTime_5 (46.8% positive rate, balanced) |
| `lgbm_ontime7.pkl` | LightGBM classifier for OnTime_7 |
| `lgbm_ontime10.pkl` | LightGBM classifier for OnTime_10 (75.4% positive) |
| `lgbm_regression.pkl` | LightGBM regressor on `target_days` (used for `pred_rtat`) |
| `xgb_ontime5.pkl` | XGBoost reference classifier (kept for comparison) |
| `xgb_regression.pkl` | XGBoost reference regressor |
| `xgb_ontime{3,5,7,10}.pkl` | XGBoost per-threshold classifiers (validated equal to LightGBM) |
| `lgbm_regression_final.pkl` | LightGBM regressor refit on all 2023-2025 (final holdout model) |

---

## `outputs/prioritization/` — Segment Prioritization

Produced by `prioritization.py` (Step 8). The business deliverables — these are the tables the stakeholder presentation references directly.

| File | Description |
|---|---|
| `priority_matrix.csv` | Market_Category × Channel segments (≥500 repairs): volume, late rates, lever indicators |
| `lever_decomposition.csv` | Per segment: primary + secondary lever, four lever scores, indicator values |
| `threshold_shift.csv` | How segment ranks shift across T = 3, 5, 7, 10 (top-10 per threshold) |
| `tier_summary.csv` | Tier-level aggregates: volume, late rates, dominant primary lever |
| `nps_validation.csv` | NPS rates by predicted-risk bucket (5 bins) — the post-hoc validation |
| `nps_by_tier.csv` | NPS rates by Market_Category for 2025 responders |
| `final_recommendation.csv` | Top-N segments with primary lever + recommendation text + cases_improvable |

The lever decomposition uses four operational levers:

| Lever | What it scores |
|---|---|
| `parts_logistics` | Parts rate × 1.15 (+2), delivery days × 1.30 (+3) |
| `engineer_deployment` | Q4 rate × 1.30 (+3 strong) or × 1.10 (+1 weak) |
| `channel_process` | `max(0, channel_risk - 3)` — DMS/DMS2/ASC=0, Premier=1, ASD=2, AE=3, SPO=4 |
| `repair_complexity` | Sealed rate × 1.30 (+2), reclaim rate × 1.30 (+1) |

The validated lever mix on the train+val cohort: **43% engineer (12) / 29% parts (8) / 18% channel (5) / 11% complexity (3)**.

---

## Regenerating these outputs

Run the six pipeline files in order. Each is self-contained and idempotent — re-running overwrites prior outputs with identical results given the same input data.

```bash
python ingestion.py
python eda.py
python feature_engineering.py
python modeling.py
python prioritization.py
python leakage_audit.py
```

See [`data/README.md`](../data/README.md) for the schema requirements and [`docs/methodology.md`](../docs/methodology.md) for the methodology details.
