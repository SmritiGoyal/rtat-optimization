# Features

This document describes the feature set used by the RTAT prediction model: **40 numeric model features**, plus one categorical feature (`parts_shipping_tier`) that is documented and leakage-audited but held out of the numeric model. Each feature is documented with:

- **Source** — which raw table or engineered upstream column it draws from
- **Formula / construction** — how the value is computed
- **Timing classification** — when the value is known relative to the prediction event (the leakage discipline)
- **Deployment safety** — whether the feature can be used in a model scored at repair intake
- **Implementation reference** — which subsection of `feature_engineering.py` owns the transformation

## Timing taxonomy

Every feature is classified into one of five timing buckets. This classification is what the leakage discipline rests on, and what the automated audit (`leakage_audit.py`) enforces.

| Class | Meaning | Deployment-safe at intake? |
|---|---|---|
| **INTAKE** | Value is determined when the repair is opened (channel, market category, engineer assignment). | ✅ Yes |
| **TRAINING** | Aggregate computed from training data only and applied via merge (tier mean RTAT, city target encoding). The current record's value doesn't enter the aggregate. | ✅ Yes |
| **POST_INTAKE** | Value becomes known shortly after intake but before completion (parts ordering decisions). | ⚠️ Conditional — depends on scoring point |
| **MEDIUM** | Assignment timing is uncertain — may be set at intake, may be set during repair (TER flag, sealed-repair flag, engineer-channel risk). | ⚠️ Confirm with operations |
| **POST_CLOSE** | Value is only available after the repair completes (actual delivery days, repair duration). | 🔴 Not deployment-safe — excluded from MODEL_FEATURES |

The pipeline includes one POST_CLOSE feature in the dataset for audit reference (`parts_order_to_arrival_days_safe`) and explicitly excludes it from `MODEL_FEATURES`. Its deployment-safe replacement is `seg_delivery_days_hist`, a segment-level historical median computed on training data only.

## Feature spec at a glance

`MODEL_FEATURES` (the leakage-reviewed list in `feature_engineering.py`) contains 41 entries split into eight groups defined in the `FEATURE_SPEC` constant. Of these, 40 are numeric and feed the trained LightGBM/XGBoost models; the one categorical-string column (`parts_shipping_tier`) is documented and audited but excluded from the numeric model:

| Group | Count | Examples |
|---|---:|---|
| Geography | 5 | `market_tier_ordinal`, `city_target_enc`, `state_target_enc` |
| Channel | 3 | `channel_risk_ordinal`, `channel_mean_rtat`, `channel_late_rate5` |
| Time | 6 | `month_of_year`, `is_weekend_close`, `is_peak_month` |
| Product | 4 | `div_mean_rtat`, `is_ter_repair`, `is_sealed_repair` |
| Engineer | 3 | `engineer_hist_mean_rtat`, `engineer_quartile`, `engineer_proxy_missing` |
| Reclaim | 6 | `has_parts_reclaim`, `is_reclaim`, `parts_complexity_score` |
| DMS | 11 | `ordered_via_dms`, `parts_line_count`, `parts_delivery_tier`, `seg_delivery_days_hist` |
| Interactions | 3 | `geo_channel_risk`, `rural_parts_flag`, `eng_channel_risk` |
| **Total in MODEL_FEATURES** | **41** | 40 numeric + 1 categorical (`parts_shipping_tier`). `parts_order_to_arrival_days_safe` is in the dataset but excluded from MODEL_FEATURES entirely. |

The two-track CORE / EXTENDED design splits the 40 numeric features by missingness pattern: CORE (31 features) is low-missingness and safe for linear models with median fill; EXTENDED (the full 40 numeric) adds DMS-dependent features (~75% missingness) for tree models that handle nulls natively.

---

## 1. Geography features

### `market_tier_ordinal`

**Source:** Raw `Market_Category` column.

**Formula:**

```python
TIER_RISK_MAP = {
    "1. Top 10": 1,   # 6.9d mean RTAT
    "2. Metro":  2,
    "3. Urban":  3,
    "4. Rural":  4,   # 13.3d mean RTAT
}
```

The map is hardcoded to lock encoding stability across data refreshes. Unknown categories fall back to `2` (Metro-level risk).

**Timing:** INTAKE — market category is known when the repair is opened.

**Rationale:** Rural repairs run ~2× longer than Top 10 markets due to engineer travel time, parts logistics, and lower density of authorized service centers. Encoding as ordinal preserves the monotone relationship the model can exploit directly.

**Implementation:** Section 6B, `add_geography_features()`.

### `tier_mean_rtat`, `tier_late_rate5`, `tier_late_rate7`

**Source:** Training-data aggregate per `Market_Category`.

**Formula:** For each tier, the mean of `target_days` over the training cohort plus the late rate at T=5 and T=7 days. Computed once on training, merged onto both train and holdout.

**Timing:** TRAINING — the current record never enters the aggregate. Unknown tiers in holdout fall back to NaN, which tree models handle natively.

**Implementation:** Section 6B.

### `city_target_enc`, `state_target_enc`

**Source:** Raw `City_` and `State_` columns.

**Formula:** Bayesian-smoothed mean encoding with prior strength `k = 30`:

```
smoothed_mean(v) = (n(v) * mean(target | v) + k * global_mean) / (n(v) + k)
```

The `k = 30` prior acts as 30 virtual observations of the global mean, pulling low-count categories toward the prior. Unseen categories in holdout fall back to the training global mean.

**Timing:** TRAINING — encoding is fit on training only, applied to holdout via map lookup.

**Rationale:** City has ~14,000 unique values; State has 56 (50 states plus DC and US territories). Without smoothing, low-count cities would memorize individual training observations and overfit. Smoothing also is a standard mitigation for target-encoding leakage.

**Implementation:** Section 6H, `smooth_target_encode()` and `add_target_encoded_features()`.

---

## 2. Channel features

### `channel_risk_ordinal`

**Source:** Raw `Channel` column.

**Formula:**

```python
CHANNEL_RISK_MAP = {
    "DMS":             1,  # 6.0d, 45.6% late
    "DMS2":            2,
    "ASC":             3,
    "Premier Partner": 4,
    "ASD":             5,
    "AE":              6,
    "SPO":             7,  # 20.8d, 86.4% late
}
```

Hardcoded to keep encoding stable. Unknown channels fall back to `3` (ASC-level risk).

**Timing:** INTAKE — channel is known when the repair is opened.

**Rationale:** Channel captures who is performing the repair. The 14.8-day RTAT gap between DMS (manufacturer's in-house dispatch) and SPO (small-parts-only channel) reflects fundamentally different operational models, not just performance differences. The ordinal encoding preserves the monotone risk ranking.

**Implementation:** Section 6C, `add_channel_features()`.

### `channel_mean_rtat`, `channel_late_rate5`, `channel_late_rate7`

Mean RTAT and late rates per channel, computed on training only, merged via `Channel` join. Same TRAINING-class timing as the geography aggregates.

**Implementation:** Section 6C.

---

## 3. Time features

All derived from `Warranty_Closed_Date`, which is stored as a `YYYYMMDD` integer in the source data (requires `pd.to_datetime(..., format='%Y%m%d')`).

| Feature | Range | Description |
|---|---|---|
| `month_of_year` | 1-12 | Calendar month of close date |
| `quarter` | 1-4 | Calendar quarter |
| `day_of_week` | 0-6 | Monday = 0 |
| `is_weekend_close` | 0/1 | 1 if closed Saturday or Sunday |
| `is_peak_month` | 0/1 | 1 if month's training-data late rate (T=5) exceeds overall training late rate |
| `month_mean_rtat` | float | Monthly mean RTAT from training data, merged via month |

**Timing:** All time features are INTAKE (date attributes) or TRAINING (`is_peak_month`, `month_mean_rtat`).

**Implementation:** Sections 6A (`fix_date_features`) and 6G (`add_seasonality_features`).

### Notes on weekend close

The original ingestion code derived weekend-flag from the raw `Warranty_Closed_Date` integer assuming `pd.to_datetime` would handle the YYYYMMDD format natively. It doesn't — generic parsing silently produces NaT, and the weekend flag ended up effectively random. The fix at the top of Step 6 (`fix_date_features`) re-parses with explicit `format='%Y%m%d'` and re-derives all date-based features from a correct timestamp. This is the kind of subtle bug that pure unit tests wouldn't catch — it only surfaced after EDA showed weekend rates that didn't make operational sense.

---

## 4. Product features

### `div_mean_rtat`, `div_late_rate5`

Mean RTAT and late rate at T=5 per `Division_Name`, computed on training only, merged onto both train and holdout. TRAINING-class timing.

**Implementation:** Section 6F.

### `is_ter_repair`, `is_sealed_repair`, `is_reclaim`, `is_same_symptom_reclaim`

Four binary flags derived from upstream columns:

| Feature | Source | Direction |
|---|---|---|
| `is_ter_repair` | `svc_ter_repair` | 1 if TER (expedited) repair. **Counterintuitively faster**: 5.5d avg vs cohort 6.8d. Initially proposed as a "complexity" feature, EDA reversed the direction — TER routing actually expedites repairs. |
| `is_sealed_repair` | `svc_sealed_repair` | 1 if sealed-system repair. ~26% of cohort, 7.5d avg (moderate). |
| `is_reclaim` | `is_reclaim_case` | 1 if a repeat visit. ~8% of cohort. |
| `is_same_symptom_reclaim` | Composite | 1 if both `is_reclaim == 1` AND `same_symptom_reclaim == 1`. ~59% of reclaim cases — the worst subset of repeat failures. |

**Timing:** MEDIUM for `is_ter_repair` and `is_sealed_repair` — assignment may be set at intake or during diagnosis. The leakage review flags both for operational confirmation. INTAKE / LOW for `is_reclaim` (prior history is always known at intake) and `is_same_symptom_reclaim` (derived from is_reclaim plus a flag set at intake).

**Implementation:** Section 6F.

---

## 5. Engineer features

### `engineer_hist_mean_rtat`

**Source:** Recomputed fold-scoped in feature engineering (`add_engineer_mean`): the per-engineer mean `target_days` is fit on the training fold only (2023-2024 for model selection; 2023-2025 for the final model) and merged onto the rows it is applied to. The current record never enters its own engineer's mean.

**Formula:** For each engineer, the mean `target_days` over their training-cohort repairs. The current record is never included in the aggregate (computation is point-in-time correct: engineer's history at the time of this repair).

**Timing:** TRAINING — historical mean is computed exclusively on training data, attached to each record via engineer ID. Holdout records get the training-period mean for their assigned engineer.

**Rationale:** This is the **single strongest feature** in the model — ~22% LightGBM importance. The 4× RTAT gap between Q1 (fastest 25%) and Q4 (slowest 25%) engineers reflects real, persistent variation in engineer effectiveness due to experience, regional knowledge, and tool/parts familiarity.

**Implementation:** Section 6, `add_engineer_mean()` (fold-scoped), consumed by `add_engineer_features()` for quartile binning. The earlier ingestion-side computation fit on the full 2023-2025 cohort and was the source of a validation-fold leak — see methodology Section 7.5.

### `engineer_quartile`

Engineer binned into quartile 1-4 based on `engineer_hist_mean_rtat`. Quartile cutpoints are computed from the training distribution and applied identically to holdout.

```
Q1 (fastest): historical mean ≤ 5.5 days
Q4 (slowest): historical mean > 11.7 days
```

Engineers with no historical record get `pd.NA` quartile, surfaced by the `engineer_proxy_missing` flag below.

**Implementation:** Section 6D, `add_engineer_features()`.

### `engineer_proxy_missing`

**Formula:** `1` if `engineer_hist_mean_rtat` is null, else `0`. Approximately 0.04% of holdout repairs have no engineer history (new hires).

**Rationale:** Tree models split on missingness directly; this explicit flag is informative for both linear models (which need it) and trees (which learn it faster).

**Implementation:** Section 6D.

---

## 6. Reclaim features

The Reclaim Parts ledger is the **ground-truth source** for whether a repair required parts. The DMS table only records parts ordered via the DMS dispatch channel (~25% of cohort), so `ordered_via_dms` is a process indicator, not a parts-required flag. `has_parts_reclaim` is the correct parts-required signal at ~71% positive.

### `has_parts_reclaim`

**Source:** Raw Reclaim `Parts_No1` through `Parts_No5` columns. `1` if any of the five part-number columns is non-null.

**Timing:** INTAKE — parts presence is recorded when the repair is opened.

**Implementation:** Upstream in ingestion.

### `parts_count_reclaim`

**Formula:** Count of non-null part columns (0-5) in the Reclaim record. Complexity proxy.

**Timing:** INTAKE.

### `parts_complexity_score`

**Formula:**

```python
parts_complexity_score = parts_count_reclaim + parts_multi_line_flag
```

Range 0-6. The `+ parts_multi_line_flag` add-on captures whether DMS multi-line ordering applied, intensifying the complexity signal when both indicators agree.

**Timing:** INTAKE (Reclaim portion) and POST_INTAKE (DMS multi-line portion). The composite is deployment-safe because both inputs are populated by mid-repair.

**Implementation:** Section 6E, `add_parts_features()`.

### `is_reclaim`, `is_same_symptom_reclaim`, `reclaim_period_days`

Repeat-visit features. `reclaim_period_days` is the days since the prior visit; null for non-reclaim repairs.

**Timing:** All INTAKE — prior visit history is known at the time of the new opening.

---

## 7. DMS features

11 features, all dependent on DMS being the parts dispatch channel. ~75% of repairs have DMS feature nulls because DMS routes only ~28% of all repairs. Tree models handle these nulls natively; linear models can't (which is why CORE excludes them).

### Process / structural

| Feature | Type | Description |
|---|---|---|
| `ordered_via_dms` | Int8 | 1 if parts ordered via DMS. ~25% positive |
| `parts_line_count` | float64 | Number of part lines ordered. Null when no DMS |
| `parts_order_qty_sum` | float64 | Total parts quantity ordered |
| `parts_multi_line_flag` | Int8 | 1 if more than one part line |
| `parts_has_arrival_flag` | Int8 | 1 if at least one part has recorded arrival |
| `parts_has_shipment_flag` | Int8 | 1 if at least one part has recorded shipment |
| `parts_shipping_tier` | categorical | OVERNIGHT / TWO_DAY / GROUND / PICKUP / OTHER / UNKNOWN. Documented and audited, but held out of the numeric model (the one non-numeric MODEL_FEATURES entry). |
| `parts_truncation_flag` | float64 | 1 if record from truncated source sheet (quality metadata) |

**Timing:** INTAKE or POST_INTAKE — all known within hours of repair opening.

### Delivery speed — the carefully-designed pair

This is the single most important design decision in DMS features. The pipeline includes two delivery features with different deployment profiles:

#### `parts_order_to_arrival_days_safe` (EDA-only, EXCLUDED from MODEL_FEATURES)

**Formula:** `parts_arrival_date - parts_order_date`, only where arrival precedes the close date (~15% of repairs).

**Timing:** POST_CLOSE conditionally — the value only exists once the parts have actually arrived, which for late repairs may be after diagnosis but before close. For deployment, this means it's unavailable at intake.

**Why it's in the dataset:** EDA reference. The audit table explicitly documents this feature as "CONDITIONAL — EXCLUDED FROM MODEL".

#### `seg_delivery_days_hist` (deployment-safe replacement, IN MODEL_FEATURES)

**Formula:** Segment-level median delivery days per `(market_tier_ordinal, channel_risk_ordinal)` combination, computed from training data only:

```python
seg_delivery = (
    train[train["parts_order_to_arrival_days_safe"].notna()]
    .groupby(["market_tier_ordinal", "channel_risk_ordinal"])
    ["parts_order_to_arrival_days_safe"]
    .median()
)
```

**Timing:** TRAINING — segment historical averages are known at intake time as soon as the market and channel are determined. No leakage.

**Rationale:** Replacing repair-level actual delivery (which leaks information) with segment-level historical median (which doesn't) preserves the predictive signal — "rural × SPO segments historically take 14 days for parts" — without any look-ahead bias. This is the single most important leakage-prevention decision in the pipeline.

**Implementation:** Section 6E.

### `parts_delivery_tier` and `parts_delivery_tier_known`

`parts_delivery_tier` bins `parts_order_to_arrival_days_safe` into 6 tiers (1=fastest, 6=slowest). Inherits the same POST_CLOSE-conditional timing.

`parts_delivery_tier_known` is the administrative flag (`1` if tier is populated, ~15% positive). Deployment-safe because it's just a presence indicator.

**Rationale:** Both columns are included so tree models can learn the joint pattern of (do we have delivery data) × (what tier was it). The latter is conditional on the former. At scoring time on a new repair, `parts_delivery_tier` is typically null and the model relies on `seg_delivery_days_hist` for the analogous segment-level signal.

**Implementation:** Section 6E, `_delivery_tier()` helper plus `add_parts_features()`.

---

## 8. Interaction features

Three engineered interactions, all deterministic functions of upstream features.

### `geo_channel_risk`

```python
geo_channel_risk = market_tier_ordinal * channel_risk_ordinal
```

Range 1-28. Top 10 × DMS = 1 (best); Rural × SPO = 28 (worst). Captures the compound risk of slow geography on slow channel.

**Timing:** INTAKE (both inputs are intake-known).

### `rural_parts_flag`

```python
rural_parts_flag = (market_tier_ordinal == 4) AND (has_parts_reclaim == 1)
```

1 if Rural tier AND parts required. ~13% of cohort. EDA identified this as the most delayed segment combination — rural geography plus parts logistics compound delays roughly multiplicatively.

**Timing:** INTAKE.

### `eng_channel_risk`

```python
eng_channel_risk = engineer_quartile * channel_risk_ordinal
```

Compound delay risk: slow engineer working on a high-risk channel. Uses `engineer_quartile.fillna(2.5)` for engineers without history (median assumption).

**Timing:** MEDIUM — engineer assignment may not always be confirmed at intake; depends on dispatch timing. Flagged in the leakage review for operational confirmation.

**Implementation:** Section 6I, `add_interaction_features()`.

---

## 9. Targets (not features)

The pipeline produces five potential targets, used for different model heads:

| Target | Type | Definition |
|---|---|---|
| `target_days` | continuous | Repair duration in days (the regression target) |
| `OnTime_3` | binary | 1 if `target_days <= 3` (strictest) |
| `OnTime_5` | binary | 1 if `target_days <= 5` (**primary classification target**) |
| `OnTime_7` | binary | 1 if `target_days <= 7` |
| `OnTime_10` | binary | 1 if `target_days <= 10` (loosest) |

Only `OnTime_5` and `target_days` are used as primary modeling targets in `modeling.py`. The other thresholds are evaluated as sensitivity analysis (Step 7).

---

## Summary table — MODEL_FEATURES

The 40 numeric features below feed the trained LightGBM/XGBoost models. The one categorical entry (`parts_shipping_tier`, #34) is part of `MODEL_FEATURES` and is leakage-audited, but is held out of the numeric model — the boosters here train on the 40 numeric columns.

| # | Feature | Group | Timing | Deployment-safe? |
|---:|---|---|---|---|
| 1 | `market_tier_ordinal` | Geography | INTAKE | ✅ |
| 2 | `tier_mean_rtat` | Geography | TRAINING | ✅ |
| 3 | `tier_late_rate5` | Geography | TRAINING | ✅ |
| 4 | `city_target_enc` | Geography | TRAINING | ✅ |
| 5 | `state_target_enc` | Geography | TRAINING | ✅ |
| 6 | `channel_risk_ordinal` | Channel | INTAKE | ✅ |
| 7 | `channel_mean_rtat` | Channel | TRAINING | ✅ |
| 8 | `channel_late_rate5` | Channel | TRAINING | ✅ |
| 9 | `month_of_year` | Time | INTAKE | ✅ |
| 10 | `quarter` | Time | INTAKE | ✅ |
| 11 | `day_of_week` | Time | INTAKE | ✅ |
| 12 | `is_weekend_close` | Time | INTAKE | ✅ |
| 13 | `is_peak_month` | Time | TRAINING | ✅ |
| 14 | `month_mean_rtat` | Time | TRAINING | ✅ |
| 15 | `div_mean_rtat` | Product | TRAINING | ✅ |
| 16 | `div_late_rate5` | Product | TRAINING | ✅ |
| 17 | `is_ter_repair` | Product | MEDIUM | ⚠ Verify |
| 18 | `is_sealed_repair` | Product | MEDIUM | ⚠ Verify |
| 19 | `engineer_hist_mean_rtat` | Engineer | TRAINING | ✅ |
| 20 | `engineer_quartile` | Engineer | TRAINING | ✅ |
| 21 | `engineer_proxy_missing` | Engineer | INTAKE | ✅ |
| 22 | `has_parts_reclaim` | Reclaim | INTAKE | ✅ |
| 23 | `parts_count_reclaim` | Reclaim | INTAKE | ✅ |
| 24 | `parts_complexity_score` | Reclaim | INTAKE/POST_INTAKE | ✅ |
| 25 | `is_reclaim` | Reclaim | INTAKE | ✅ |
| 26 | `is_same_symptom_reclaim` | Reclaim | INTAKE | ✅ |
| 27 | `reclaim_period_days` | Reclaim | INTAKE | ✅ |
| 28 | `ordered_via_dms` | DMS | POST_INTAKE | ✅ |
| 29 | `parts_line_count` | DMS | POST_INTAKE | ✅ |
| 30 | `parts_order_qty_sum` | DMS | POST_INTAKE | ✅ |
| 31 | `parts_multi_line_flag` | DMS | POST_INTAKE | ✅ |
| 32 | `parts_has_arrival_flag` | DMS | POST_INTAKE | ✅ |
| 33 | `parts_has_shipment_flag` | DMS | POST_INTAKE | ✅ |
| 34 | `parts_shipping_tier` | DMS | POST_INTAKE | ✅ (categorical — audited, held out of the numeric model) |
| 35 | `parts_delivery_tier` | DMS | POST_CLOSE conditional | ⚠ Conditional |
| 36 | `parts_delivery_tier_known` | DMS | POST_INTAKE | ✅ |
| 37 | `seg_delivery_days_hist` | DMS | TRAINING | ✅ |
| 38 | `parts_truncation_flag` | DMS | INTAKE | ✅ |
| 39 | `geo_channel_risk` | Interactions | INTAKE | ✅ |
| 40 | `rural_parts_flag` | Interactions | INTAKE | ✅ |
| 41 | `eng_channel_risk` | Interactions | MEDIUM | ⚠ Verify |

The 40 numeric model features are rows 1-33 and 35-41 (every row above except the categorical `parts_shipping_tier` at #34).

**Plus, in the dataset but EXCLUDED from MODEL_FEATURES:**

| - | `parts_order_to_arrival_days_safe` | DMS | POST_CLOSE | 🔴 No |

---

## References to other documentation

- **Production code:** `feature_engineering.py` — every transformation here is documented in the corresponding subsection (6A through 6O).
- **Leakage audit:** `leakage_audit.py` runs five tests on every feature to catch the subtle leakage patterns; results in `outputs/features/leakage_review.csv`.
- Note: the 5-test audit does not detect validation-fold encoder leaks; see methodology Section 7.5.
- **Methodology writeup:** `docs/methodology.md` — full discussion of the cohort filter, target encoding smoothing, engineer historical mean discipline, two-track CORE/EXTENDED feature design, and hyperparameter selection.
- **Schema:** `data/README.md` — full source-data schema documentation with channel glossary and cohort filter rules.
