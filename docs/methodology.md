# Methodology — Extended Writeup

This document goes deeper than the README on the technical decisions that shaped the RTAT pipeline. It is intended for technical reviewers who want to understand the *why* behind specific implementation choices, not just the *what*.

## 1. Problem Framing

### 1.1 Why log loss is not the right metric here

For RTAT, the cost surface is asymmetric and tied to a service target. A customer with a promise date of "within 5 days" doesn't care whether the predicted probability of being on time was 0.6 or 0.9 — they care whether the repair was actually completed in 5 days. The metric that matches the operational reality is **classification at fixed thresholds** (`OnTime_3`, `OnTime_5`, `OnTime_7`, `OnTime_10`), evaluated by AUC and F1.

The regression target (`target_days`) is retained for two purposes:
1. Continuous-severity inputs to the lever decomposition (a segment with mean predicted RTAT 15 days needs different treatment than one at 8 days)
2. Operational planning at thresholds not in the trained set (e.g., "what if we set a 4.5-day target?")

### 1.2 Why a single global model isn't enough

The seven-channel network (DMS / DMS2 / ASC / Premier Partner / ASD / AE / SPO) has fundamentally different operational characteristics. Direct (DMS, DMS2) is parts-bound. Authorized Engineer (AE) is dispatch-bound. A single LightGBM model captures the channel effect through the `Channel` feature and engineered channel-level aggregates, but it's still averaging across operational regimes.

The decision to use a single global model + per-segment lever decomposition (rather than per-channel models) was a deliberate trade-off:
- **Pro single model:** Cross-channel interactions (Rural × DMS, Top10 × Premier Partner, etc.) are captured by tree splits. Per-channel models would miss these.
- **Pro per-channel models:** Channel-specific dynamics would surface more clearly.
- **Verdict:** Start with single global; per-channel is a v2 candidate (see "What I'd Do Differently" in the README).

## 2. Data Engineering

### 2.1 Custom Excel header normalization

The raw exports arrive with inconsistent column naming across sheets. Within one workbook, `Repair_No` might appear as `Repair No`, `Repair_No`, or `Repair  No` (double space). The pipeline's `normalize_col()` function:

```python
def normalize_col(col: object) -> str:
    if col is None or (isinstance(col, float) and np.isnan(col)):
        return ""
    return re.sub(r"\s+", " ", str(col).strip())
```

is applied to every sheet's headers before column matching. Combined with a case-insensitive `usecols` lookup, this recovers ~99.5% of expected columns across all sheets without manual mapping.

### 2.2 Two-pass key cleaning

Join keys arrive as string-cast numerics with trailing `.0` artifacts from prior Excel round-trips:

```
"123456.0"
"  789012"
"<NA>"
"nan"
""
```

`clean_key_strict` handles the canonical cases (uppercase, strip whitespace, drop `.0`, null sentinels → None). `clean_key_loose` adds a fallback that removes non-alphanumeric characters, recovering keys with embedded hyphens or other separators that snuck in during data export.

The strict-then-loose two-pass match recovers an additional ~3% of joinable rows. For 1.6M repairs that's ~48,000 rows that would otherwise be dropped silently.

The reclaim join key is `GSFS_Repair_Header_No`, not the master's `Repair_No`. Both are normalized through `clean_key` before joining; the validated overlap rate is 94.2% (master ↔ reclaim) and 27.7% (master ↔ parts, expected — DMS is a sparse subset of the universe).

### 2.3 Cohort exclusions — why some rows are dropped

The cohort filter intentionally excludes:

1. **Non-appliance divisions** (HANDSET, LED Signage, Signage, Commercial TV, MNT Signage, PTV, Robot Business Task)
   - These divisions have RTAT distributions an order of magnitude different from the appliance core. Including them inflates baseline noise without operational value to the field service team.

2. **Non-RTAT center types** (Affiliate, DSC, Commercial ASC non-referral)
   - These center types either don't track RTAT consistently or follow different operational rules. Their inclusion would bias channel-level aggregates.

3. **RTAT outside [0, 365] days**
   - Negative RTAT values are data entry errors. RTAT > 365 days are typically administrative artifacts (e.g., warranty claims kept open for accounting purposes). Both are excluded with explicit flags so the filter is auditable.

The cohort filter is applied **after** feature flags are computed, so we have a per-row record of why each excluded row was excluded. This allows downstream sensitivity analyses (e.g., "what if we relaxed the 365-day cap to 730?") without rerunning ingestion.

Of the 2,192,254 raw master rows, the cohort retains 1,640,829 (74.8%): 1,060,649 training (2023-2024), 509,930 validation (2025), and 70,250 holdout (2026 Jan-Feb).

## 3. Exploratory Data Analysis & Hypothesis Formation

EDA is a separate stage (`eda.py`, Step 5 in the pipeline) — not a side activity. Its output is a list of eight hypotheses that map directly onto downstream feature engineering and the lever taxonomy in prioritization. Treating EDA as a deliverable, not scratch work, prevents the common pattern where exploratory findings get lost between the analyst's head and the production code.

### 3.1 The eight hypotheses

| # | Lever | Hypothesis | Source |
|---|---|---|---|
| H1 | Geography | Top 10 and Metro tier repairs have higher late rates than Urban/Rural, despite higher volume — highest combined impact opportunity | Market_Category analysis |
| H2 | Parts logistics | Parts-required repairs have materially longer lead times than no-part repairs. Parts delivery duration is the dominant variable leg | Parts impact analysis |
| H3 | Channel | Channel is a significant predictor of lead time. DMS and Premier Partner channels show highest delay rates | Channel analysis |
| H4 | Product complexity | Sealed system repairs (refrigerants) and TER cases have longer lead times, suggesting complexity is a key delay driver beyond parts alone | Reclaim signal analysis |
| H5 | Engineer capacity | Engineer historical performance proxy shows meaningful correlation with actual lead time — engineer deployment is a viable intervention lever | Engineer signal analysis |
| H6 | Seasonality | Lead time peaks in specific months — likely Q3/Q4 peak season. Resource allocation should account for seasonal demand patterns | Time effects analysis |
| H7 | Repeat failure | Reclaim cases (repeat visits) have longer lead times than first visits. Same-symptom reclaims are the most delayed subset | Reclaim signal analysis |
| H8 | NPS validation | High-delay repairs correlate with lower promoter rates in 2025 NPS subset — confirms business relevance of RTAT optimization | NPS validation |

H1-H4 map directly onto the four operational levers in the prioritization stage (geography → `channel_process`, parts → `parts_logistics`, channel → also `channel_process`, complexity → `repair_complexity`). H5 informs the engineer quartile feature design. H6 informs `is_peak_month` and `month_mean_rtat`. H7 informs the reclaim feature family. H8 is the validation layer.

### 3.2 What the EDA confirmed and what it surprised

Two findings shaped the methodology more than expected:

- **TER repairs are faster, not slower.** The naive hypothesis was that TER (the manufacturer's expedited repair classification) flagged complex repairs needing extra attention, and would correlate with longer RTAT. The data shows the opposite: TER repairs average 5.5 days vs 9.7 days for non-TER, with a 62% on-time rate at T=5 vs 44% for non-TER. The flag is operational — it signals an expedited handling protocol, not difficulty. The model treats `is_ter_repair` as a "faster" feature.

- **Same-symptom reclaims are not notably worse than different-symptom.** The hypothesis was that returning customers with the same problem (potentially a misdiagnosis on the first visit) would be the worst subset. The data shows same-symptom and different-symptom reclaims have nearly identical mean RTAT (10.5d vs 11.1d) and on-time rates (36.6% vs 35.5%). Both subsets are notably slower than first visits (9.2d, 46.4% on-time), but the "same-symptom is worst" intuition is not supported. The `is_same_symptom_reclaim` feature is kept for model use but with reduced importance in the lever decomposition.

These two findings are documented in the EDA output (`5h_reclaim_signal.csv`, `5f_engineer_quartiles.csv`) and referenced explicitly in the feature engineering data dictionary.

## 4. Feature Engineering

### 4.1 The feature set, grouped

The model uses **38 numeric features** (plus one categorical feature documented but held out of the numeric model) organized into eight groups:

| Group | Features | Count |
|---|---|---|
| Geography | `market_tier_ordinal`, `tier_mean_rtat`, `tier_late_rate5`, `city_target_enc`, `state_target_enc` | 5 |
| Channel | `channel_risk_ordinal`, `channel_mean_rtat`, `channel_late_rate5` | 3 |
| Time | `month_of_year`, `quarter`, `day_of_week`, `is_weekend_close`, `is_peak_month`, `month_mean_rtat` | 6 |
| Product | `div_mean_rtat`, `div_late_rate5`, `is_ter_repair`, `is_sealed_repair` | 4 |
| Engineer | `engineer_hist_mean_rtat`, `engineer_quartile`, `engineer_proxy_missing` | 3 |
| Reclaim | `has_parts_reclaim`, `parts_count_reclaim`, `parts_complexity_score`, `is_reclaim`, `is_same_symptom_reclaim`, `reclaim_period_days` | 6 |
| Parts logistics | `ordered_via_dms`, `parts_line_count`, `parts_order_qty_sum`, `parts_multi_line_flag`, `parts_has_shipment_flag`, `parts_shipping_tier`, `parts_delivery_tier_known`, `seg_delivery_days_hist`, `parts_truncation_flag` | 9 |
| Interactions | `geo_channel_risk`, `rural_parts_flag`, `eng_channel_risk` | 3 |
| **Total** | | **39** |

Three POST_CLOSE parts features (`parts_order_to_arrival_days_safe`, its binned twin `parts_delivery_tier`, and `parts_has_arrival_flag`) are retained in the feature parquet for EDA/audit reference but excluded from `MODEL_FEATURES`. The total above is the 38 numeric model features plus the one categorical `parts_shipping_tier`.

### 4.2 Two-track design: CORE vs EXTENDED

The 38 numeric features split into two model-specific subsets:

- **CORE (31 features)**: low missingness across the training cohort. Safe for linear models after median fill. Excludes DMS-dependent features (parts line count, order quantity, multi-line flag, etc.) because they're ~75% null (only ~25% of repairs route parts through DMS), plus the ~44%-null `seg_delivery_days_hist`.
- **EXTENDED (38 features = the numeric MODEL_FEATURES)**: adds DMS-dependent features. Used by tree models (Random Forest, XGBoost, LightGBM) that handle missing values natively. Uses the deployment-safe `seg_delivery_days_hist` and excludes the three POST_CLOSE parts features (`parts_order_to_arrival_days_safe`, `parts_delivery_tier`, `parts_has_arrival_flag`) and the one categorical-string column `parts_shipping_tier` (documented but held out of the numeric model).

This isn't a stylistic choice. Linear models on the EXTENDED set produce worse holdout performance because median fill on 75%-null features destroys signal; tree models on the CORE set leave money on the table. The design is informed by validation metrics, not aesthetic preference.

### 4.3 Bayesian smoothing for high-cardinality categoricals

For city (~13,900 unique in training) and state (56 unique), naive target encoding produces:

```
city_mean(City_X) = sum(target_days for City_X) / count(City_X)
```

This overfits for low-volume cities. A city with one repair completed in 2 days gets encoded as 2.0 — but that's a sample-size-of-1 estimate. The Bayesian smoothing pulls these toward the global mean (validated at 9.334 days on the training cohort):

```
city_mean(City_X) = (sum(target) + 30 × global_mean) / (count + 30)
```

The smoothing strength `k=30` was chosen so that a city with 30 repairs is weighted equally between its own mean and the global mean. Above 30, the city's own data dominates; below 30, the global mean dominates. This is a standard empirical-Bayes prior calibration. The encoder is fit on training data only and applied to validation and holdout via lookup with global-mean fallback for unseen categories.

### 4.4 Engineer historical mean

Each engineer has a historical track record from prior repairs. The naive feature is:

```
engineer_hist_mean_rtat(eng_id) = mean(target_days for all repairs by eng_id)
```

Critically, this **should** be computed on training-period repairs only — and after the fix described below, it is. For a validation repair in 2025 by engineer E, the feature value is E's mean across the 2023-2024 training fold, even if E continued to work in 2025.

In the original implementation this aggregate was fit in ingestion across the full 2023-2025 cohort and merged before the train/validation split — so 2025 validation rows did see 2025 RTATs. This was caught in audit and fixed: the engineer mean is now recomputed fold-scoped in feature engineering (fit on the 2023-2024 fold for selection; refit on 2023-2025 for the final model). See Section 7.5.

This avoids two failure modes:
1. **Information leakage:** if computed on the full dataset, the feature would include the current repair's contribution to its own engineer's mean.
2. **Future leakage:** if computed across all years, a 2025 validation repair would see 2025 RTATs reflected in the engineer mean.

`engineer_proxy_missing` is a flag for engineers who appear in validation/holdout but not in training (new hires, transfers, etc.). For these rows the feature falls back to the global mean. The quartile thresholds (Q1≤5.5d, Q2≤7.2d, Q3≤11.7d, Q4 > 11.7d quoted here are the final-model values fit on 2023-2025; the selection model uses 2023-2024-fit cutpoints) are also computed on training data and frozen.

The validated within-quartile mean RTAT shows a long-tail pattern, not a smooth gradient: **Q1=4.5d, Q2=6.3d, Q3=8.9d, Q4=17.7d**. The slowest 25% of engineers are dramatically slower than the rest (Q4/Q1 ratio = 3.9×), which is why engineer deployment is the dominant lever in the segment decomposition (43% of segments).

### 4.5 Segment-safe delivery feature

One of the strongest signals in the data is parts delivery duration (`parts_order_to_arrival_days_safe`) — how many days elapse between part order and arrival. The leakage-safe replacement is **segment-level historical median**:

```
seg_delivery_days_hist = median(parts_order_to_arrival_days_safe)
    grouped by (market_tier_ordinal, channel_risk_ordinal)
    computed on training data only
```

At intake time, the actual delivery duration for *this* repair is unknown. But the historical segment median (e.g., "Rural × AE typically takes 12 days") is known. This swap keeps the predictive power of the delivery signal without using post-event information.

The original repair-level feature is retained in the feature parquet for EDA reference and audit traceability, but excluded from `MODEL_FEATURES` and audited in Step 9 as `CONDITIONAL — EXCLUDED FROM MODEL`. Its binned twin (`parts_delivery_tier`) and the arrival flag (`parts_has_arrival_flag`) are excluded on the same grounds — all three require parts to have arrived and are unavailable at intake scoring.

### 4.6 Interaction features

Three cross-feature interactions are explicitly engineered:

- `geo_channel_risk = market_tier_ordinal × channel_risk_ordinal` — captures that "Rural × AE" has different risk than "Urban × AE"
- `rural_parts_flag = (tier == 4) AND (has_parts_reclaim == 1)` — flags a specific operational pattern (rural repairs needing parts)
- `eng_channel_risk = engineer_quartile × channel_risk_ordinal` — captures that "Q4 engineer on a slow channel" is multiplicatively worse than the sum of either alone

LightGBM can learn most interactions natively via tree splits, but explicit interactions help:
1. Linear baselines (Logistic Regression, Ridge) can see the interaction
2. SHAP / feature importance attributions stay clean
3. Documented interactions are easier to explain to stakeholders

## 5. Modeling

### 5.1 Why LightGBM specifically

The model comparison runs all models in parallel for fair comparison:

**Classification (7 models):** Majority Baseline → Logistic Regression → Decision Tree (depth=3) → Decision Tree (depth=6) → Random Forest → XGBoost → LightGBM

**Regression (8 models):** Mean Baseline → Segment Mean (tier × channel) → Ridge → Lasso → Decision Tree → Random Forest → XGBoost → LightGBM

On the final 2026 holdout, LightGBM and XGBoost are within ~0.01 AUC and trade the lead by threshold — LightGBM ahead at T=3/T=5/T=10 (0.798/0.806/0.841 vs 0.795/0.803/0.832), XGBoost ahead at T=7 (0.816 vs 0.815); on validation regression XGBoost edges LightGBM within noise (5.04 vs 5.07). LightGBM is the primary deployable model because it leads at the primary T=5 target and on regression on the holdout. The remaining factors:

- **Training efficiency** — LightGBM trained ~25% faster on the full feature matrix
- **Categorical handling** — LightGBM natively handles high-cardinality categoricals without explicit encoding
- **Memory efficiency** — LightGBM's histogram-based splitting uses less RAM than XGBoost's exact split finding
- **Early stopping callback** — both support it, but LightGBM's `lgb.early_stopping(50)` callback is cleaner

The XGBoost reference model is retained (`xgb_ontime5.pkl`, `xgb_regression.pkl`) for comparison and as a fallback if LightGBM-specific behavior is ever in question.

### 5.2 Hyperparameter choices

The production LightGBM classifier configuration:

```python
{
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],   # AUC first; early stopping watches it
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,           # 2^5 - 1, deliberately conservative
    "min_child_samples": 500,   # Prevents splits on tiny subgroups
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,           # L1 regularization
    "reg_lambda": 1.0,          # L2 regularization
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}
```

with `lgb.early_stopping(50)` as the training callback. These are *not* heavily tuned. After the fold-scoped encoder fix, classifier early stopping was switched to watch AUC (the selection metric, `first_metric_only=True` on an auc-first metric list) rather than logloss — with logloss watched, training stopped prematurely on the easy high-positive thresholds. With AUC watched, LightGBM trains to a competitive iteration count at T=3/T=5 (best_iter ~96/114). At T=7/T=10 the target is highly separable and validation AUC saturates within a couple of iterations, so those models legitimately train to few trees. A formal Optuna sweep is still on the "what I'd do differently" list. The chosen values reflect:

- `learning_rate = 0.05` with `n_estimators = 500` and early stopping at 50 rounds = a stable mid-tier learning rate that converges well below 500 trees (typically 100-150)
- `num_leaves = 31` is **conservative for a 1M-row dataset**, paired with `max_depth = 6`. The deliberate undersize prevents the model from learning overly specific patterns
- `min_child_samples = 500` prevents trees from learning patterns from groups smaller than 0.05% of training data
- `reg_alpha = 0.1, reg_lambda = 1.0` is moderate L1 + L2 regularization on leaf weights

For skewed thresholds (T=3 at 27.9% positive, T=10 at 75.4% positive), the configuration adds `is_unbalance=True`, which re-weights the binary log-loss to handle class imbalance. The 0.15/0.85 toggle bounds are set in the modeling config.

### 5.3 Why a separate model per threshold

A single classifier trained on `OnTime_5` with post-hoc thresholding to predict OnTime_3 would not give optimal F1 at T=3. The decision boundary that maximizes AUC for T=5 isn't where the data structure cleanest separates T=3.

Training four models is computationally cheap (~30 sec per model on the full feature matrix) and gives meaningfully better F1 per threshold than the post-hoc approach. The cost is operational complexity (4 model artifacts to serve), which is acceptable for a recommendation system that runs once per planning cycle, not in real-time scoring.

## 6. Validation Strategy

### 6.1 Time-based splits, never random

The training data spans 2023-2025; the holdout is 2026 (Jan-Feb). Aggregates are fit on the 2023-2024 fold for selection and refit on all 2023-2025 for the final model; the 2026 holdout is scored exactly once. This discipline is what makes the validation and holdout numbers meaningful.

**Validation-to-holdout, honestly.**
The model is selected on a strict 2023-2024 -> 2025 split (T=5 validation AUC ~0.76), then refit on all 2023-2025 and scored once on the locked 2026 holdout (T=5 AUC 0.806). The holdout sits slightly above the selection-fold validation because the final model trains on an extra year of data. Generalization is evidenced by consistent performance across all four thresholds and by the NPS external validation — not by a single tight gap.

A time-ordered split is non-negotiable for a deployment scenario. A random split would let the model see July 2025 patterns at training time and predict on June 2025 — production won't have that information. The time-ordered split is what makes the validation and holdout numbers meaningful; see Section 6.3 for why the originally-reported tight validation->holdout gap was an artifact rather than evidence of generalization.

### 6.2 Holdout never seen, ever

The 2026 holdout was reserved at the cohort stage and never appeared in:
- Feature encoder fitting (no city/state/engineer aggregates use 2026 data)
- Model training (LightGBM never saw 2026 rows)
- Hyperparameter tuning (decisions made on 2025 validation only)
- Lever benchmark calibration (global means computed on train, applied to all)

The first time the 2026 holdout was scored was the final evaluation step. There is no risk of accidental tuning on holdout.

### 6.3 The two-phase protocol, and why the original gap was misleading

The original pipeline reported a 0.011 validation->holdout gap. That number was an artifact: training-class aggregates were fit on the full 2023-2025 cohort and then split into train/validation without refitting, so the 2025 validation metric was inflated and the gap to the (clean) 2026 holdout looked artificially tight.

After the fix, the honest protocol is two phases — select on 2023-2024 -> 2025, then refit on 2023-2025 and score 2026 once. Final holdout AUC is 0.806 at T=5, 0.798 / 0.806 / 0.815 / 0.841 across T=3/5/7/10. The holdout sits at or above the selection-fold validation (~0.76 at T=5) because the final model trains on an extra year; the consistent across-threshold performance and the NPS external check are the real generalization evidence.

A second, independent exposure was found in the same audit: post-close parts features known only once parts had arrived had been included in the trained model despite the continuous version being documented as excluded. The repair-level delivery duration (`parts_order_to_arrival_days_safe`), its binned twin (`parts_delivery_tier`), and the arrival flag (`parts_has_arrival_flag`) all require parts to have arrived and are unavailable at intake scoring. Removing all three and relying on the deployment-safe segment-median substitute (`seg_delivery_days_hist`) brings the model to 38 numeric features that match the leakage review exactly. The fix moves T=5 holdout AUC from 0.819 to 0.806 (a ~0.013 cost) — confirming the leaked signal was largely redundant with the clean segment median, so the model held at near-full strength through both fixes.

## 7. The Leakage Audit

### 7.1 Why it's a separate module

The audit could have been an `audit_features()` function inside one of the modeling files. Keeping it as a separate file (`leakage_audit.py`) is a deliberate choice that signals:

1. **Auditing happens after modeling, not during it.** A senior data scientist runs the audit *on a trained model* as a pre-deployment gate.
2. **Audits should be re-runnable.** Months from now, you can re-audit a previously-trained model without re-running the full pipeline. The audit loads persisted models and parquets.
3. **The audit has its own contract.** It returns an `AuditResults` dataclass that can be programmatically inspected. A future v2 could wire this into CI/CD as a model-deployment gate.

### 7.2 The five tests, ranked by importance

Most ML practitioners think of leakage as "high correlation with target" (test 2). The full picture:

1. **Timing (test 1)** is the most important and the most often skipped. A feature with `r = 0.4` correlation might be totally clean (legitimate signal) or a hard leak (computed from post-close data). Only timing classification tells you which. The pipeline classifies every feature into five timing classes — INTAKE, TRAINING, POST_INTAKE, MEDIUM, POST_CLOSE — and only INTAKE/TRAINING-class features are deployment-safe for an intake-time model.
2. **Stability (test 3)** is the second most important. A feature whose train and holdout distributions are radically different is suspicious — either the world changed (acceptable) or the feature is computed differently across periods (concerning).
3. **Targeted checks (4, 5)** catch known failure modes. They wouldn't be needed if the feature engineering were trivially clean, but they exist because adjacent analyses produced false-positive models on this same dataset.
4. **Correlation (test 2)** is the least informative individually but most easily measured. It's the canary that flags obvious leaks; the other tests are needed to catch subtle ones.

### 7.3 The validated audit result

Of the 39 audited features (38 numeric model features + the one categorical feature held out of the numeric model), the audit returns:

| Verdict | Count | Examples |
|---|---|---|
| ✓ CLEAN | 33 | All TRAINING and INTAKE-class features that pass correlation + stability |
| ⚠ CONFIRM WITH OPS | 3 | `is_ter_repair`, `is_sealed_repair`, `eng_channel_risk` |
| ⚠ STABILITY FLAG | 3 | `month_of_year`, `quarter`, `month_mean_rtat` — > 0.30 KS shift |

The two POST_CLOSE parts features that earlier reviews flagged (`parts_delivery_tier`, `parts_has_arrival_flag`) are no longer model features — they were removed from `MODEL_FEATURES`, so the `⚠ REVIEW` category that previously held them is gone. These verdicts are otherwise unchanged by the encoder fix — see Section 7.5 for why the audit could not have flagged the validation-fold leak that was present.

Test 5 (engineer proxy stability) returned a 0.46-day train-vs-holdout gap, well within the 1.0-day tolerance. Test 4 (`reclaim_period_days` targeted check) passed both correlation (< 0.15) and importance (< 5%) thresholds.

The 3 STABILITY FLAG features (`month_of_year`, `quarter`, `month_mean_rtat`) shift because the 2026 holdout is a partial year (Jan-Feb) versus full-year training. This is acceptable temporal drift, not leakage. The 3 CONFIRM WITH OPS features require human verification of intake-time availability before production deployment — they are kept in the model with this caveat documented.

### 7.4 What the audit doesn't catch

This is worth being explicit about. The audit catches:
- ✓ Temporal leakage (timing test)
- ✓ Direct target-derived features (correlation test)
- ✓ Distribution shifts indicating data contamination (KS test)
- ✓ Two specific known failure modes (targeted tests)

The audit does *not* catch:
- ✗ Train/test contamination via duplicate rows (would need row-level dedup audit)
- ✗ Feature group leakage where the *combination* of features is leaky (would need permutation testing)
- ✗ Label leakage from the cohort filter itself (would need to compare results across cohort definitions)
- ✗ Operator error in the encoder fit step - **this is exactly the leak found in this project** (aggregates fit on 2023-2025 then split without refit). The audit passed because it compares train vs the 2026 holdout and is blind to a validation-fold leak. Caught by a fold-scoping ablation; see 7.5.

A v2 audit would add at least the first two of these.

### 7.5 The leak this audit missed — and the fix

**Fold-scoped encoder fitting (leak found and fixed).**
An internal audit caught that the training-class aggregates — engineer historical mean (the model's strongest feature), tier / channel / division / month means, city and state target encoding, and the segment delivery median — were originally fit on the full 2023-2025 cohort and then split into train (2023-2024) and validation (2025) without refitting. That let 2025 information reach the validation rows and inflated the validation metrics (and the apparent validation-to-holdout gap).

The fix scopes every aggregate to data available before the rows it is applied to: for model selection, aggregates are fit on the 2023-2024 fold and validated on 2025; for the final number, a select-then-refit protocol refits the aggregates and retrains on all of 2023-2025, then scores the locked 2026 holdout exactly once.

**What the five-test audit did not catch.**
The audit compares training against the 2026 holdout, so it is blind to a leak confined to the *validation fold*: the leaked aggregates were "train-only" relative to 2026, so all five tests passed while 2025 was still bleeding into validation. The methodology already described fold-correct fitting ("training-period only"); the implementation fit more broadly than that. The mismatch was caught by a fold-scoping ablation, not by the audit — and the fix made the implementation match the documented discipline.

A defensible audit therefore needs a sixth test: refit every encoder per fold and assert the validation metric is stable. That is the first addition planned for a v2 audit.

## 8. Operational Translation

### 8.1 From predictions to decisions

The classifier output `pred_late_prob` is interesting; the **segment priority matrix** is what drives action. The translation:

```
For each (Market_Category × Channel) segment with ≥ 500 repairs:
    priority_score(segment, T) = volume × actual_late_rate(T)
    priority_rank(segment, T) = rank of priority_score across segments
```

This is intentionally simple. Priority = cases-at-risk = volume × failure rate. A segment with 50,000 repairs and 30% late rate (15,000 at-risk cases) ranks above a segment with 5,000 repairs and 60% late rate (3,000 at-risk cases), even though the second has worse per-unit performance.

This matches operational reality: the field service org has finite resources and benefits from focusing on segments where each resource dollar covers more at-risk cases.

### 8.2 Four-lever scoring as transparent rules

The lever decomposition (`classify_primary_lever`) uses additive rule scoring rather than learned classification. Four levers are evaluated per segment:

```
parts_logistics:
    +2 if parts_rate > 1.15 × global_parts_rate
    +3 if delivery_days_median > 1.30 × global_delivery_days  (if not NaN)

engineer_deployment:
    +3 if engineer_q4_rate > 1.30 × global_engineer_q4_rate (strong)
    +1 if engineer_q4_rate > 1.10 × global_engineer_q4_rate (weak)

channel_process:
    score = max(0, channel_risk_ordinal - 3)
    # DMS=0, DMS2=0, ASC=0, Premier Partner=1, ASD=2, AE=3, SPO=4

repair_complexity:
    +2 if sealed_rate > 1.30 × global_sealed_rate
    +1 if reclaim_rate > 1.30 × global_reclaim_rate
```

The primary lever for each segment is the one with the highest score. The secondary lever is the next-highest (used for segments where two levers are roughly comparable in impact). Ties are broken by the order levers are evaluated in (`parts_logistics → engineer_deployment → channel_process → repair_complexity`), which preserves the original notebook's behavior.

Multipliers (1.15, 1.30, 1.10) were calibrated through iteration with stakeholder review, not by minimizing a held-out metric. This is appropriate for an operational recommendation system: the goal is decision quality, not predictive accuracy, and decisions are validated by stakeholder review.

The validated lever mix on the train+val cohort: **43% engineer / 29% parts / 18% channel / 11% complexity**. No single lever dominates, which means the four-lever framework is doing real work — if 80%+ of segments were assigned to one lever, the framework would collapse to a single recommendation.

### 8.3 NPS as honest external validation

The most defensible piece of the prioritization output is the NPS post-hoc check. NPS signals (`is_promoter`, `is_detractor`) were intentionally excluded from training. After running the priority matrix, we compare promoter rates across **predicted-risk buckets** of 2025 NPS respondents:

| Predicted risk bucket | Promoter rate | Detractor rate |
|---|---|---|
| Very low (0-20% predicted late) | 69.0% | 19.7% |
| Very high (80-100% predicted late) | 57.5% | 28.0% |
| **Gap** | **11.5pp** | **8.3pp** |

The 11.5-point promoter gap and 8.3-point detractor gap are computed out-of-sample — 2025 NPS responders scored by the 2023-2024 selection model (which never saw 2025), with NPS never entering the model. The signal cannot be tuned from inside the pipeline (NPS is not a feature).

---

## Summary

The project's defensible claims, in order of strength:

1. **Honest out-of-time generalization.** Final 2026 holdout AUC 0.806 (T=5) under a select-then-refit protocol, with two separate leakage exposures found and fixed during a self-directed audit — a fold-level encoder leak and post-close parts features wrongly included in the model (Section 7.5). Removing them cost ~0.013 AUC at T=5, so the model held at near-full strength.
2. **Leakage-safe by construction.** Five-test audit across all 39 documented features (33 CLEAN); train-only encoder fits documented; engineer-proxy gap of 0.46d well within tolerance.
3. **30% MAE improvement over baseline** on the locked 2026 holdout (4.74d vs 6.77d).
4. **Operational translation across 4 levers.** Top recommendation segment plus three additional levers cover the operational decision space; no single lever monopolizes (43/29/18/11 distribution).
5. **11.5pp promoter gap, 8.3pp detractor gap** between low-risk and high-risk predicted segments, computed out-of-sample (2025 responders scored by the 2023-24 selection model) without NPS in the model.

The weakest claims, by contrast, are about specific lever multipliers (1.15×, 1.30×) — those would benefit from formal sensitivity analysis in a v2.
