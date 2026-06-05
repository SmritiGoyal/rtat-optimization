"""
feature_engineering.py
======================
Step 6: Feature Engineering for the RTAT pipeline.

Builds the model feature matrix from the integrated master table produced
by ``ingestion.py``. All encodings and aggregates are fit fold-scoped — on
the 2023-2024 training fold for model selection, then refit on all 2023-2025
for the final model — and applied to the validation rows and the 2026 holdout
via merge, never re-fit on the data the model is scored against. This
fold-scoping is the single most important leakage-prevention discipline in
the pipeline.

(An earlier version fit these aggregates on the full 2023-2025 cohort before
the train/validation split, leaking 2025 into validation; that is the leak
this fold-scoped design fixes.)

Per-fold feature chain — in ``_engineer_all()``, run once per pass
(Pass 1: fit on 2023-2024, apply to train + 2025 val; Pass 2: fit on
2023-2024, apply to the 2026 holdout):
    6A: Fix weekend close (Warranty_Closed_Date is YYYYMMDD int)
    6B: Geography features  (Market_Category ordinal, tier stats)
    6C: Channel features    (Channel ordinal, channel stats)
    6D: Engineer features   (fold-scoped historical mean, quartile, missing proxy flag)
    6E: Parts features      (delivery tier, seg_delivery_days_hist, complexity)
    6F: Product features    (division stats, repair-type flags)
    6G: Seasonality         (month aggregates, peak month flag)
    6H: Target encoding     (city/state with Bayesian smoothing k=30)
    6I: Interactions        (geo × channel, rural × parts, eng × channel)

Assembly, audit, and save — in ``run_feature_engineering()``, after the
two passes are reassembled:
    6J: Final feature spec  (FEATURE_SPEC dict, MODEL_FEATURES list)
    6K: Missingness summary
    6L: Leakage review
    6M: Data dictionary
    6N: Save feature tables
    6O: Final summary

Two entry points:
    run_feature_engineering()        — fold-scoped selection build (fit on
                                       2023-2024, apply to 2025 val + 2026)
    run_feature_engineering_final()  — final build (fit on all 2023-2025)

Output artifacts under ``outputs/features/``:
    feature_train.parquet            (2023-2025, aggregates fit on 2023-2024 fold)
    feature_holdout.parquet          (2026, aggregates fit on 2023-2024 fold)
    feature_train_final.parquet      (2023-2025, aggregates fit on all 2023-2025)
    feature_holdout_final.parquet    (2026, aggregates fit on all 2023-2025)
    feature_spec.csv
    leakage_review.csv
    data_dictionary.csv
    missingness_summary.csv
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the feature engineering stage."""
    # Directories
    interim_dir: Path = PROJECT_ROOT / "outputs" / "interim"
    feature_dir: Path = PROJECT_ROOT / "outputs" / "features"

    # OnTime thresholds we model (full set from ingestion is 1..10;
    # this 4-threshold subset is what downstream stages consume)
    ontime_targets: tuple[int, ...] = (3, 5, 7, 10)

    # Smoothing prior strength for high-cardinality target encoding
    smoothing_k: int = 30

    # Random seed (used by downstream; kept here for reference)
    random_state: int = 42


# Market_Category ordinal — ordered by observed mean RTAT in training data
# (Top10 fastest 6.9d → Rural slowest 13.3d). Hardcoded to lock the
# encoding in case observed ordering changes on a future data refresh.
TIER_RISK_MAP: dict[str, int] = {
    "1. Top 10": 1,
    "2. Metro":  2,
    "3. Urban":  3,
    "4. Rural":  4,
}

# Channel ordinal — ordered by observed late rate at T=5 in training.
# DMS fastest (45.6% late) → SPO slowest (86.4% late). Hardcoded so the
# encoding is stable across data refreshes and reproducible across runs.
CHANNEL_RISK_MAP: dict[str, int] = {
    "DMS":             1,
    "DMS2":            2,
    "ASC":             3,
    "Premier Partner": 4,
    "ASD":             5,
    "AE":              6,
    "SPO":             7,
}


# =====================================================================
# SECTION 6A: WEEKEND CLOSE FIX
# =====================================================================

def fix_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive weekend / month / quarter / day-of-week from YYYYMMDD int.

    The raw ``Warranty_Closed_Date`` is stored as a YYYYMMDD integer in
    the source data. Generic ``pd.to_datetime`` fails on these; we must
    use explicit ``format='%Y%m%d'``. This function fixes the weekend
    flag and re-derives every date-based feature from a correctly parsed
    timestamp.

    Operates on a copy.
    """
    df = df.copy()
    closed_dt = pd.to_datetime(
        df["Warranty_Closed_Date"].astype(str).str.strip(),
        format="%Y%m%d", errors="coerce",
    )
    df["is_weekend_close"] = closed_dt.dt.dayofweek.isin([5, 6]).astype("Int8")
    df["month_of_year"] = closed_dt.dt.month.astype("Int8")
    df["quarter"] = closed_dt.dt.quarter.astype("Int8")
    df["day_of_week"] = closed_dt.dt.dayofweek.astype("Int8")  # 0=Mon
    return df


# =====================================================================
# SECTION 6B: GEOGRAPHY FEATURES
# =====================================================================

def add_geography_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add market tier ordinal + tier aggregates (mean RTAT, late rates).

    Tier statistics are computed on training data only and merged onto
    both train and holdout. Unknown tier values fall back to ordinal 2
    (Metro-level risk).

    Returns:
        (train, holdout, tier_stats) — both DataFrames with new columns
        plus the fitted tier_stats table for documentation.
    """
    # Apply ordinal map
    for df in (train, holdout):
        df["market_tier_ordinal"] = (
            df["Market_Category"].map(TIER_RISK_MAP)
            .fillna(2)
            .astype("Int8")
        )

    # Mean RTAT + late rate per tier — training only
    tier_stats = (
        train.groupby("Market_Category")["target_days"]
        .agg(
            tier_mean_rtat="mean",
            tier_late_rate5=lambda x: (x > 5).mean(),
            tier_late_rate7=lambda x: (x > 7).mean(),
        )
        .round(4)
        .reset_index()
    )

    train = train.merge(tier_stats, on="Market_Category", how="left")
    holdout = holdout.merge(tier_stats, on="Market_Category", how="left")
    return train, holdout, tier_stats


# =====================================================================
# SECTION 6C: CHANNEL FEATURES
# =====================================================================

def add_channel_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add channel ordinal + channel aggregates from training data.

    Unknown channels fall back to ordinal 3 (ASC-level risk).
    """
    channel_stats = (
        train.groupby("Channel")["target_days"]
        .agg(
            channel_mean_rtat="mean",
            channel_late_rate5=lambda x: (x > 5).mean(),
            channel_late_rate7=lambda x: (x > 7).mean(),
            channel_n="count",
        )
        .round(4)
    )

    for df in (train, holdout):
        df["channel_risk_ordinal"] = (
            df["Channel"].map(CHANNEL_RISK_MAP)
            .fillna(3)
            .astype("Int8")
        )

    channel_encoded = channel_stats.reset_index()
    train = train.merge(channel_encoded, on="Channel", how="left")
    holdout = holdout.merge(channel_encoded, on="Channel", how="left")
    return train, holdout, channel_stats


# =====================================================================
# SECTION 6D: ENGINEER FEATURES
# =====================================================================

def add_engineer_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Add engineer quartile + missing-proxy flag.

    Quartile thresholds are computed from training's ``engineer_hist_mean_rtat``
    distribution (which is recomputed fold-scoped by ``add_engineer_mean()``
    immediately upstream in ``_engineer_all`` — no longer the ingestion
    Step 4C value). Engineers with no historical record get pd.NA quartile,
    surfaced by the ``engineer_proxy_missing`` flag for the model.

    Returns:
        (train, holdout, quartile_thresholds) — DataFrames plus the
        Q1/Q2/Q3 cut points for documentation.
    """
    eng_proxy = train["engineer_hist_mean_rtat"].dropna()
    q1, q2, q3 = eng_proxy.quantile([0.25, 0.50, 0.75])

    def assign_eng_quartile(val):
        """Map an engineer's historical mean RTAT to quartile 1..4."""
        if pd.isna(val):
            return pd.NA
        if val <= q1:
            return 1  # fastest
        if val <= q2:
            return 2
        if val <= q3:
            return 3
        return 4  # slowest

    for df in (train, holdout):
        df["engineer_quartile"] = (
            df["engineer_hist_mean_rtat"]
            .apply(assign_eng_quartile)
            .astype("Int8")
        )
        df["engineer_proxy_missing"] = (
            df["engineer_hist_mean_rtat"].isna()
        ).astype("Int8")

    return train, holdout, {"q1": float(q1), "q2": float(q2), "q3": float(q3)}


def add_engineer_mean(
        train: pd.DataFrame,
        holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Recompute engineer_hist_mean_rtat on the FIT cohort (first arg) only.

    Mirrors ingestion Step 4C exactly
    (``groupby("SVC_Engineer_Code")["target_days"].mean()``) but fits on the
    rows passed here — which the caller scopes to the appropriate fit cohort
    (the 2023-2024 fold for model selection; all 2023-2025 for the final
    build). Drops any engineer_hist_mean_rtat carried in from ingestion (fit
    on the full 2023-2025 cohort) and replaces it, so the quartile binning in
    add_engineer_features() reads the fold-correct value.

    Engineers absent from the fit cohort get NaN, surfaced downstream by
    engineer_proxy_missing.
    """
    for df in (train, holdout):
        df.drop(columns=["engineer_hist_mean_rtat"], errors="ignore", inplace=True)

    eng_mean = (
        train.groupby("SVC_Engineer_Code")["target_days"]
        .mean()
        .rename("engineer_hist_mean_rtat")
    )
    train = train.merge(eng_mean, on="SVC_Engineer_Code", how="left")
    holdout = holdout.merge(eng_mean, on="SVC_Engineer_Code", how="left")
    return train, holdout, eng_mean


# =====================================================================
# SECTION 6E: PARTS FEATURES
# =====================================================================

def _delivery_tier(days) -> int | pd._libs.missing.NAType:
    """Bin actual parts delivery duration into 6 speed tiers.

    Derived from EDA's monotonic finding: 0-1d=60% ontime → 5d+=~3% ontime.

        1: 0-1 days   (fast)
        2: 1-2 days
        3: 2-3 days
        4: 3-5 days
        5: 5-7 days
        6: 7+ days    (very slow)
    """
    if pd.isna(days):
        return pd.NA
    if days <= 1:
        return 1
    if days <= 2:
        return 2
    if days <= 3:
        return 3
    if days <= 5:
        return 4
    if days <= 7:
        return 5
    return 6


def add_parts_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add parts delivery tier + segment-safe historical delivery + complexity.

    Three feature families:

    1. ``parts_delivery_tier`` — speed tier from actual repair-level
       order→arrival days. Available only when arrival precedes close
       (~15% of repairs). Captured alongside a ``_known`` admin flag.

    2. ``seg_delivery_days_hist`` — segment-level (market_tier × channel)
       median delivery days, computed on training data and applied via
       merge. **Deployment-safe replacement** for the repair-level
       feature, because segment historical averages are known at intake
       time. This is THE leakage-safe feature designed for live scoring.

    3. ``parts_complexity_score`` — composite: parts_count_reclaim +
       parts_multi_line_flag. Range 0-6, captures repair complexity.

    Also adds ``has_dms_delivery_data`` administrative flag.

    Returns:
        (train, holdout, seg_delivery) — the seg_delivery table is
        returned for documentation.
    """
    # Repair-level delivery tier
    for df in (train, holdout):
        df["parts_delivery_tier"] = (
            df["parts_order_to_arrival_days_safe"]
            .apply(_delivery_tier)
            .astype("Int8")
        )
        df["parts_delivery_tier_known"] = (
            df["parts_delivery_tier"].notna()
        ).astype("Int8")

    # Segment-level historical median (deployment-safe alternative)
    # — replaces parts_order_to_arrival_days_safe in MODEL_FEATURES
    seg_delivery = (
        train[train["parts_order_to_arrival_days_safe"].notna()]
        .groupby(["market_tier_ordinal", "channel_risk_ordinal"])
        ["parts_order_to_arrival_days_safe"]
        .median()
        .rename("seg_delivery_days_hist")
        .reset_index()
    )

    train = train.merge(
        seg_delivery, on=["market_tier_ordinal", "channel_risk_ordinal"], how="left",
    )
    holdout = holdout.merge(
        seg_delivery, on=["market_tier_ordinal", "channel_risk_ordinal"], how="left",
    )

    # Composite complexity score
    for df in (train, holdout):
        df["parts_complexity_score"] = (
            df["parts_count_reclaim"].astype(float).fillna(0)
            + df["parts_multi_line_flag"].astype(float).fillna(0)
        ).astype("float32")
        df["has_dms_delivery_data"] = (
            df["parts_order_to_arrival_days_safe"].notna()
        ).astype("Int8")

    return train, holdout, seg_delivery


# =====================================================================
# SECTION 6F: PRODUCT / COMPLEXITY FEATURES
# =====================================================================

def add_product_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add division aggregates and repair-type binary flags.

    Repair-type flag direction (from EDA):
        is_ter_repair    — TER repairs are FASTER (5.5d avg, expedited)
        is_sealed_repair — Sealed system, moderate (7.5d avg)
        is_reclaim       — Repeat visit, 8.2% of cohort

    Both is_ter_repair and is_sealed_repair carry MEDIUM deployment-safe
    risk: their assignment timing may be post-intake. Surfaced in the
    leakage review.
    """
    div_stats = (
        train.groupby("Division_Name")["target_days"]
        .agg(
            div_mean_rtat="mean",
            div_late_rate5=lambda x: (x > 5).mean(),
            div_n="count",
        )
        .round(4)
    )
    div_encoded = div_stats.reset_index()
    train = train.merge(div_encoded, on="Division_Name", how="left")
    holdout = holdout.merge(div_encoded, on="Division_Name", how="left")

    # Repair-type flags
    for df in (train, holdout):
        df["is_ter_repair"] = df["svc_ter_repair"].fillna(0).astype("Int8")
        df["is_sealed_repair"] = df["svc_sealed_repair"].fillna(0).astype("Int8")
        df["is_reclaim"] = df["is_reclaim_case"].fillna(0).astype("Int8")
        # Reclaim severity: same symptom = worst subset (58.8% of reclaims)
        df["is_same_symptom_reclaim"] = (
            (df["is_reclaim"] == 1)
            & (df["same_symptom_reclaim"] == 1)
        ).astype("Int8")

    return train, holdout, div_stats


# =====================================================================
# SECTION 6G: SEASONALITY FEATURES
# =====================================================================

def add_seasonality_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Add monthly mean RTAT and a peak-month flag.

    A month is "peak" if its training-data late rate at T=5 exceeds the
    overall training late rate. The set of peak months is learned from
    training and applied identically to holdout.

    Returns:
        (train, holdout, peak_months) — peak_months list returned for
        documentation/audit.
    """
    month_stats = (
        train.groupby("month_of_year")["target_days"]
        .agg(
            month_mean_rtat="mean",
            month_late_rate5=lambda x: (x > 5).mean(),
        )
        .round(4)
    )

    avg_late = train["target_days"].gt(5).mean()
    peak_months: list[int] = month_stats[
        month_stats["month_late_rate5"] > avg_late
    ].index.tolist()

    month_encoded = month_stats.reset_index()
    train = train.merge(month_encoded, on="month_of_year", how="left")
    holdout = holdout.merge(month_encoded, on="month_of_year", how="left")

    for df in (train, holdout):
        df["is_peak_month"] = df["month_of_year"].isin(peak_months).astype("Int8")

    return train, holdout, peak_months


# =====================================================================
# SECTION 6H: TARGET ENCODING — HIGH-CARDINALITY FIELDS
# =====================================================================

def smooth_target_encode(
    train_df: pd.DataFrame,
    col: str,
    target_col: str,
    k: int,
    global_mean: float,
) -> pd.Series:
    """Compute Bayesian-smoothed mean encoding on training data.

    For each category ``v``::

        smoothed_mean(v) = (n(v) * mean(target | v) + k * global_mean) / (n(v) + k)

    The prior strength ``k`` (= 30) acts as virtual observations of the
    global mean. Low-count categories are pulled toward the global mean,
    high-count categories are dominated by their own mean. This both
    prevents overfitting to rare categories AND is a standard mitigation
    for target encoding leakage.

    Returns a mapping Series safe to apply to both train and holdout.
    Unseen categories in holdout get the global mean fallback at apply
    time, not at fit time.
    """
    stats = (
        train_df.groupby(col)[target_col]
        .agg(n="count", mean_enc="mean")
    )
    stats["smoothed"] = (
        (stats["n"] * stats["mean_enc"] + k * global_mean)
        / (stats["n"] + k)
    )
    return stats["smoothed"]


def add_target_encoded_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply smoothed mean encoding to City_ and State_ (high-cardinality).

    Returns:
        (train, holdout, encoding_diagnostics) — diagnostics dict has
        n_cities, n_states, and global_mean for documentation.
    """
    global_mean = float(train["target_days"].mean())

    city_enc = smooth_target_encode(
        train, "City_", "target_days", cfg.smoothing_k, global_mean,
    )
    train["city_target_enc"] = (
        train["City_"].map(city_enc).fillna(global_mean).astype("float32")
    )
    holdout["city_target_enc"] = (
        holdout["City_"].map(city_enc).fillna(global_mean).astype("float32")
    )

    state_enc = smooth_target_encode(
        train, "State_", "target_days", cfg.smoothing_k, global_mean,
    )
    train["state_target_enc"] = (
        train["State_"].map(state_enc).fillna(global_mean).astype("float32")
    )
    holdout["state_target_enc"] = (
        holdout["State_"].map(state_enc).fillna(global_mean).astype("float32")
    )

    diagnostics = {
        "n_cities": int(train["City_"].nunique()),
        "n_states": int(train["State_"].nunique()),
        "global_mean": global_mean,
    }
    return train, holdout, diagnostics


# =====================================================================
# SECTION 6I: INTERACTION FEATURES
# =====================================================================

def add_interaction_features(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add three engineered interaction features.

    1. ``geo_channel_risk``: market_tier_ordinal × channel_risk_ordinal.
       Range 1-28. Rural × SPO = 28 (worst), Top10 × DMS = 1 (best).
    2. ``rural_parts_flag``: 1 if Rural tier AND has_parts_reclaim.
       Most-delayed segment interaction.
    3. ``eng_channel_risk``: engineer_quartile × channel_risk_ordinal.
       Compound delay risk for slow engineer on high-risk channel.
    """
    for df in (train, holdout):
        df["geo_channel_risk"] = (
            df["market_tier_ordinal"].astype(float)
            * df["channel_risk_ordinal"].astype(float)
        ).astype("float32")

        df["rural_parts_flag"] = (
            (df["market_tier_ordinal"] == 4)
            & (df["has_parts_reclaim"] == 1)
        ).astype("Int8")

        eng_q = df["engineer_quartile"].astype(float).fillna(2.5)
        ch_r = df["channel_risk_ordinal"].astype(float).fillna(3)
        df["eng_channel_risk"] = (eng_q * ch_r).astype("float32")

    return train, holdout


# =====================================================================
# SECTION 6J: FEATURE SPEC + MODEL_FEATURES LIST
# =====================================================================

FEATURE_SPEC: dict[str, list[str]] = {
    # --- ALWAYS AVAILABLE ---
    "geography": [
        "market_tier_ordinal",
        "tier_mean_rtat",
        "tier_late_rate5",
        "city_target_enc",
        "state_target_enc",
    ],
    "channel": [
        "channel_risk_ordinal",
        "channel_mean_rtat",
        "channel_late_rate5",
    ],
    "time": [
        "month_of_year",
        "quarter",
        "day_of_week",
        "is_weekend_close",
        "is_peak_month",
        "month_mean_rtat",
    ],
    "product": [
        "div_mean_rtat",
        "div_late_rate5",
        "is_ter_repair",
        "is_sealed_repair",
    ],
    "engineer": [
        "engineer_hist_mean_rtat",
        "engineer_quartile",
        "engineer_proxy_missing",
    ],
    # --- RECLAIM DEPENDENT (~94% coverage) ---
    "reclaim": [
        "has_parts_reclaim",
        "parts_count_reclaim",
        "parts_complexity_score",
        "is_reclaim",
        "is_same_symptom_reclaim",
        "reclaim_period_days",
    ],
    # --- DMS DEPENDENT (~28% coverage) ---
    "dms": [
        "ordered_via_dms",
        "parts_line_count",
        "parts_order_qty_sum",
        "parts_multi_line_flag",
        "parts_has_arrival_flag",
        "parts_has_shipment_flag",
        "parts_shipping_tier",
        "parts_delivery_tier",
        "parts_delivery_tier_known",
        "parts_order_to_arrival_days_safe",  # EDA/conditional reference only
        "seg_delivery_days_hist",            # deployment-safe segment aggregate
        "parts_truncation_flag",
    ],
    # --- INTERACTIONS ---
    "interactions": [
        "geo_channel_risk",
        "rural_parts_flag",
        "eng_channel_risk",
    ],
    # --- TARGETS (not features) ---
    "targets": [
        "target_days",
        "OnTime_3", "OnTime_5", "OnTime_7", "OnTime_10",
    ],
    # --- META (not features) ---
    "meta": [
        "repair_no_clean", "source_year", "is_holdout",
        "flag_cohort", "flag_excluded_division",
        "flag_excluded_center_type",
    ],
}


def build_model_features_list() -> list[str]:
    """Assemble MODEL_FEATURES from FEATURE_SPEC, excluding the EDA-only feature.

    ``parts_order_to_arrival_days_safe`` is kept in FEATURE_SPEC for
    documentation but excluded from active model features because it
    requires post-event arrival data. ``seg_delivery_days_hist`` is its
    deployment-safe replacement.
    """
    feats = (
        FEATURE_SPEC["geography"]
        + FEATURE_SPEC["channel"]
        + FEATURE_SPEC["time"]
        + FEATURE_SPEC["product"]
        + FEATURE_SPEC["engineer"]
        + FEATURE_SPEC["reclaim"]
        + FEATURE_SPEC["dms"]
        + FEATURE_SPEC["interactions"]
    )
    # Exclude the EDA-only conditional feature
    return [f for f in feats if f != "parts_order_to_arrival_days_safe"]


MODEL_FEATURES: list[str] = build_model_features_list()


# =====================================================================
# SECTION 6K: MISSINGNESS SUMMARY
# =====================================================================

def build_missingness_summary(
    train: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """Build a per-feature missingness table including group and dtype.

    Used both as a CSV deliverable and as a fail-safe check that every
    feature in ``feature_list`` exists in the training frame.
    """
    rows = []
    for feat in feature_list:
        if feat not in train.columns:
            continue
        n_miss = int(train[feat].isna().sum())
        group = next(
            (g for g, fs in FEATURE_SPEC.items() if feat in fs), "unknown",
        )
        rows.append({
            "feature": feat,
            "group": group,
            "pct_missing": round(n_miss / len(train), 4),
            "dtype": str(train[feat].dtype),
            "n_unique": int(train[feat].nunique()),
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


# =====================================================================
# SECTION 6L: LEAKAGE REVIEW
# =====================================================================

LEAKAGE_REVIEW_ROWS: list[dict] = [
    {"feature": "engineer_hist_mean_rtat", "risk": "LOW",
     "reason": "Recomputed fold-scoped (fit on the 2023-2024 training fold for "
               "selection; on 2023-2025 for the final model), merged onto the rows "
               "it serves — its own fit never includes the validation or holdout "
               "years. Engineer history exists at prediction time, equivalent to a "
               "credit-score history."},
    {"feature": "tier_mean_rtat / tier_late_rate5", "risk": "LOW",
     "reason": "Training aggregate. Applied to holdout via merge."},
    {"feature": "channel_mean_rtat / channel_late_rate5", "risk": "LOW",
     "reason": "Training aggregate. Applied to holdout via merge."},
    {"feature": "city_target_enc", "risk": "LOW",
     "reason": "Smoothed target encoding (k=30) on training only. Holdout unseen "
               "cities get global mean fallback."},
    {"feature": "state_target_enc", "risk": "LOW",
     "reason": "Same as city_target_enc. Lower cardinality."},
    {"feature": "month_mean_rtat", "risk": "LOW",
     "reason": "Monthly aggregate from training. Month is known at prediction time."},
    {"feature": "seg_delivery_days_hist", "risk": "LOW",
     "reason": "Segment-level median delivery days computed from training data only "
               "(market tier × channel). Deployment-safe replacement for repair-level "
               "parts_order_to_arrival_days_safe. Historical segment average is "
               "known at intake."},
    {"feature": "parts_order_to_arrival_days_safe", "risk": "CONDITIONAL — EXCLUDED FROM MODEL",
     "reason": "Repair-level actual delivery duration. Requires parts arrival to "
               "have occurred — not available at real-time scoring. Retained in "
               "dataset for EDA and audit documentation only. Replaced by "
               "seg_delivery_days_hist in MODEL_FEATURES."},
    {"feature": "has_parts_reclaim", "risk": "LOW",
     "reason": "Parts presence recorded at intake. Known before repair completes."},
    {"feature": "is_ter_repair", "risk": "MEDIUM",
     "reason": "TER designation timing uncertain — may be set post-diagnosis. "
               "Confirm with operations before production deployment."},
    {"feature": "is_sealed_repair", "risk": "MEDIUM",
     "reason": "Sealed system flag may be assigned after diagnosis rather than "
               "at intake. Confirm timing with operations. High importance (5.6%) "
               "— warrants explicit validation."},
    {"feature": "is_reclaim", "risk": "LOW",
     "reason": "Prior visit history is known at intake."},
    {"feature": "div_mean_rtat", "risk": "LOW",
     "reason": "Division-level training aggregate. Division known at intake."},
    {"feature": "geo_channel_risk", "risk": "LOW",
     "reason": "Product of two known-at-intake features."},
    {"feature": "eng_channel_risk", "risk": "MEDIUM",
     "reason": "Engineer assignment may not always be confirmed at intake — "
               "depends on dispatch timing. Monitor."},
    {"feature": "parts_arrival_dt_max (raw)", "risk": "HIGH — EXCLUDED",
     "reason": "Raw arrival timestamp is after repair completion. Excluded from "
               "modeling entirely."},
    {"feature": "repair_duration_days (Reclaim)", "risk": "HIGH — EXCLUDED",
     "reason": "Requires repair end timestamp — same as target. Excluded from "
               "modeling entirely."},
]


def build_leakage_review() -> pd.DataFrame:
    """Return the leakage review table as a DataFrame ready for CSV export."""
    return pd.DataFrame(LEAKAGE_REVIEW_ROWS)


# =====================================================================
# SECTION 6M: DATA DICTIONARY
# =====================================================================

DATA_DICTIONARY_ROWS: list[dict] = [
    # Geography
    {"feature": "market_tier_ordinal", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Market_Category 1–4 ordinal. Rural=4 (worst), Top10=1 (best). "
                    "Ordered by observed mean RTAT."},
    {"feature": "tier_mean_rtat", "source": "Master", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Mean RTAT per tier from training. 6.9d–13.3d."},
    {"feature": "tier_late_rate5", "source": "Master", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Late rate at T=5 per tier from training. 43%–68%."},
    {"feature": "city_target_enc", "source": "Master", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Smoothed mean RTAT per city (k=30). Range 4.0–26.2d. "
                    "Unseen cities get global mean."},
    {"feature": "state_target_enc", "source": "Master", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Smoothed mean RTAT per state (k=30)."},
    # Channel
    {"feature": "channel_risk_ordinal", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Channel ranked 1–7 by mean RTAT. DMS=1 (6.0d fastest) → "
                    "SPO=7 (20.8d worst)."},
    {"feature": "channel_mean_rtat", "source": "Master", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Mean RTAT per channel from training."},
    {"feature": "channel_late_rate5", "source": "Master", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Late rate at T=5 per channel from training."},
    # Time
    {"feature": "month_of_year", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Month 1–12 from Warranty_Closed_Date (YYYYMMDD format)."},
    {"feature": "quarter", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Quarter 1–4."},
    {"feature": "day_of_week", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Day of week 0–6 (Mon=0)."},
    {"feature": "is_weekend_close", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if closed Saturday or Sunday."},
    {"feature": "is_peak_month", "source": "Master", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 for months above average late rate on training data."},
    {"feature": "month_mean_rtat", "source": "Master", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Monthly mean RTAT from training. Seasonal signal."},
    # Product
    {"feature": "div_mean_rtat", "source": "Reclaim", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Division mean RTAT from training."},
    {"feature": "div_late_rate5", "source": "Reclaim", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Division late rate at T=5 from training."},
    {"feature": "is_ter_repair", "source": "Reclaim", "type": "Int8",
     "model_use": "feature", "deployment_safe": "MEDIUM — verify timing",
     "description": "1 if TER repair. EDA: TER = FASTER (5.5d avg) — expedited "
                    "protocol. Flag direction reversed vs proposal."},
    {"feature": "is_sealed_repair", "source": "Reclaim", "type": "Int8",
     "model_use": "feature", "deployment_safe": "MEDIUM — verify timing",
     "description": "1 if sealed system repair. ~26% of cohort. Avg 7.5d. "
                    "Raw value = 'Sealed Repair' string."},
    # Engineer
    {"feature": "engineer_hist_mean_rtat", "source": "Engineered", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Historical mean RTAT per engineer, recomputed fold-scoped "
                    "(2023-2024 fold for selection; 2023-2025 for the final model). "
                    "Strongest single feature (~22% LightGBM importance). 4x "
                    "RTAT gap Q1 vs Q4. Fully known at prediction time."},
    {"feature": "engineer_quartile", "source": "Engineered", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Engineer binned 1–4 from hist_mean_rtat. Q1≤5.5d, Q4≥11.7d "
                    "(final-model fit; the 2023-2024 selection-fold cutpoints "
                    "differ slightly)."},
    {"feature": "engineer_proxy_missing", "source": "Engineered", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if engineer_hist_mean_rtat is null (~0.04% of repairs)."},
    # Reclaim
    {"feature": "has_parts_reclaim", "source": "Reclaim", "type": "Int8",
     "model_use": "primary_feature", "deployment_safe": "YES",
     "description": "True parts-required flag from Reclaim Parts_No1-5. ~71% positive. "
                    "Ground truth parts indicator — use this NOT ordered_via_dms."},
    {"feature": "parts_count_reclaim", "source": "Reclaim", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Parts columns populated in Reclaim (0–5). Complexity proxy."},
    {"feature": "parts_complexity_score", "source": "Engineered", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Composite: parts_count_reclaim + parts_multi_line_flag. "
                    "Range 0–6. Mean ~1.59."},
    {"feature": "is_reclaim", "source": "Reclaim", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if repeat visit. ~8% of cohort. Prior history known at intake."},
    {"feature": "is_same_symptom_reclaim", "source": "Reclaim", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if reclaim + same symptom. ~59% of reclaim cases. "
                    "Worst repeat failure subset."},
    {"feature": "reclaim_period_days", "source": "Reclaim", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Days since prior visit for reclaim cases. Null for non-reclaim. "
                    "Historical — known at intake."},
    # DMS
    {"feature": "ordered_via_dms", "source": "DMS", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if parts ordered via DMS channel. ~25% of cohort. "
                    "Process indicator — NOT a parts-required flag."},
    {"feature": "parts_line_count", "source": "DMS", "type": "float64",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Part lines ordered via DMS. ~75% null = no DMS record."},
    {"feature": "parts_order_qty_sum", "source": "DMS", "type": "float64",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Total parts quantity ordered. Structural null pattern."},
    {"feature": "parts_multi_line_flag", "source": "DMS", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if more than one part line ordered."},
    {"feature": "parts_has_arrival_flag", "source": "DMS", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if at least one part has recorded arrival."},
    {"feature": "parts_has_shipment_flag", "source": "DMS", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if at least one part has recorded shipment."},
    {"feature": "parts_shipping_tier", "source": "Engineered", "type": "categorical",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Simplified shipping: OVERNIGHT/TWO_DAY/GROUND/PICKUP/OTHER/UNKNOWN."},
    {"feature": "parts_delivery_tier", "source": "DMS", "type": "Int8",
     "model_use": "feature", "deployment_safe": "CONDITIONAL",
     "description": "Delivery speed tier 1–6 from actual order→arrival days. "
                    "~15% populated. Timing-conditional — see seg_delivery_days_hist."},
    {"feature": "parts_delivery_tier_known", "source": "DMS", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if parts_delivery_tier is populated. ~15% positive. "
                    "Administrative flag — no leakage."},
    {"feature": "seg_delivery_days_hist", "source": "DMS+Engineered", "type": "float",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "Segment-level MEDIAN parts delivery days (market tier × channel). "
                    "Computed from training data only. Deployment-safe replacement "
                    "for parts_order_to_arrival_days_safe. Known at intake as a "
                    "historical segment average."},
    {"feature": "parts_order_to_arrival_days_safe", "source": "DMS+Master",
     "type": "float32",
     "model_use": "eda_reference — EXCLUDED FROM MODEL_FEATURES",
     "deployment_safe": "CONDITIONAL — excluded from modeling",
     "description": "Actual repair-level delivery days where arrival < close. "
                    "~15% non-null. Timing-conditional for live scoring. "
                    "Replaced by seg_delivery_days_hist in MODEL_FEATURES."},
    {"feature": "parts_truncation_flag", "source": "DMS", "type": "float64",
     "model_use": "quality_flag", "deployment_safe": "YES",
     "description": "1 if from truncated source sheet. Quality metadata."},
    # Interactions
    {"feature": "geo_channel_risk", "source": "Engineered", "type": "float32",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "market_tier_ordinal × channel_risk_ordinal. Range 1–28. "
                    "Rural × SPO = 28 (worst), Top10 × DMS = 1 (best)."},
    {"feature": "rural_parts_flag", "source": "Engineered", "type": "Int8",
     "model_use": "feature", "deployment_safe": "YES",
     "description": "1 if Rural tier AND has_parts_reclaim=1. ~13% of cohort. "
                    "Most delayed segment interaction."},
    {"feature": "eng_channel_risk", "source": "Engineered", "type": "float32",
     "model_use": "feature", "deployment_safe": "MEDIUM",
     "description": "engineer_quartile × channel_risk_ordinal. Compound delay risk. "
                    "Engineer assignment timing may vary by channel — confirm "
                    "before production."},
]


def build_data_dictionary() -> pd.DataFrame:
    """Return the data dictionary as a DataFrame ready for CSV export."""
    return pd.DataFrame(DATA_DICTIONARY_ROWS)


# =====================================================================
# SECTION 6N: PARQUET SAVERS
# =====================================================================

def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize object columns so PyArrow can serialize them.

    Object columns occasionally contain "nan"/"None"/"<NA>"/"NaT" strings
    from upstream casts. Returns these as pd.NA.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = (out[col].astype(str)
                        .replace({"None": "", "nan": "", "<NA>": "", "NaT": ""})
                        .replace("", pd.NA))
    return out


def build_save_columns(df: pd.DataFrame) -> list[str]:
    """Pick the subset of columns to persist on feature parquets.

    Includes meta, every MODEL_FEATURES column, the EDA-only
    ``parts_order_to_arrival_days_safe`` for audit reference, and all
    target columns. Deduplicated while preserving order. Columns absent
    from ``df`` are silently dropped.
    """
    save_cols = (
        FEATURE_SPEC["meta"]
        + MODEL_FEATURES
        + ["parts_order_to_arrival_days_safe"]
        + FEATURE_SPEC["targets"]
    )
    return list(dict.fromkeys(c for c in save_cols if c in df.columns))


def build_feature_spec_table(train: pd.DataFrame) -> pd.DataFrame:
    """Build the long-form feature spec table for CSV export.

    One row per (feature, group) pair across all FEATURE_SPEC groups,
    annotated with availability in train, missingness rate, dtype, and
    whether the feature is in MODEL_FEATURES.
    """
    spec_rows: list[dict] = []
    for group, feats in FEATURE_SPEC.items():
        for feat in feats:
            spec_rows.append({
                "feature": feat,
                "group": group,
                "in_train": feat in train.columns,
                "pct_missing": round(train[feat].isna().mean(), 4)
                               if feat in train.columns else None,
                "dtype": str(train[feat].dtype) if feat in train.columns else "N/A",
                "in_model": feat in MODEL_FEATURES,
            })
    return pd.DataFrame(spec_rows)


# =====================================================================
# MAIN ORCHESTRATION
# =====================================================================

def _engineer_all(
        train: pd.DataFrame,
        holdout: pd.DataFrame,
        cfg,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the per-fold feature chain (subsections 6A-6I), FITTING every
    aggregate on `train` and applying to both `train` and the second frame.
    Returns (train, holdout, stats), where `stats` collects the fitted
    tables for logging/documentation.

    Called twice by run_feature_engineering: once to fit on the 2023-2024
    fold and transform train + 2025 validation, once to fit on the same fold
    and transform the 2026 holdout. run_feature_engineering_final calls it
    once, fitting on all 2023-2025.
    """
    train = fix_date_features(train)  # 6A
    holdout = fix_date_features(holdout)

    train, holdout, tier_stats = add_geography_features(train, holdout)  # 6B
    train, holdout, channel_stats = add_channel_features(train, holdout)  # 6C
    train, holdout, _eng_mean = add_engineer_mean(train, holdout)  # NEW
    train, holdout, eng_q_thresholds = add_engineer_features(train, holdout)  # 6D
    train, holdout, seg_delivery = add_parts_features(train, holdout)  # 6E
    train, holdout, div_stats = add_product_features(train, holdout)  # 6F
    train, holdout, peak_months = add_seasonality_features(train, holdout)  # 6G
    train, holdout, enc_diag = add_target_encoded_features(train, holdout, cfg)  # 6H
    train, holdout = add_interaction_features(train, holdout)  # 6I

    stats = {
        "tier_stats": tier_stats,
        "channel_stats": channel_stats,
        "eng_q_thresholds": eng_q_thresholds,
        "seg_delivery": seg_delivery,
        "div_stats": div_stats,
        "peak_months": peak_months,
        "enc_diag": enc_diag,
    }
    return train, holdout, stats


def run_feature_engineering(cfg: "FeatureConfig | None" = None) -> dict:
    """Run the full Step 6 feature engineering end-to-end (fold-scoped).

    Reads ``master_train.parquet`` (2023-2025) and ``master_holdout.parquet``
    (2026) from ``cfg.interim_dir``. Every training-class aggregate is fit on
    the 2023-2024 TRAIN FOLD only, then applied as a pure lookup to the 2025
    validation rows and the 2026 holdout — closing the leak where aggregates
    were previously fit on the full 2023-2025 cohort and bled 2025 into the
    validation rows.

    Writes:
        feature_train.parquet
        feature_holdout.parquet
        feature_spec.csv
        leakage_review.csv
        data_dictionary.csv
        missingness_summary.csv
    """
    cfg = cfg or FeatureConfig()
    cfg.feature_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load full train cohort (2023-2025) + locked holdout (2026) -----
    logger.info("Loading train and holdout from %s", cfg.interim_dir)
    train_full = pd.read_parquet(cfg.interim_dir / "master_train.parquet")
    holdout = pd.read_parquet(cfg.interim_dir / "master_holdout.parquet")
    logger.info("Train_full(2023-2025): %s rows | Holdout(2026): %s rows",
                f"{len(train_full):,}", f"{len(holdout):,}")

    # ----- Split the train cohort into the 2023-24 FIT fold and 2025 val -----
    TRAIN_YEARS = (2023, 2024)
    VAL_YEAR = 2025
    assert "source_year" in train_full.columns, "source_year required for fold split"
    assert holdout["source_year"].eq(2026).all(), "holdout must be 2026 only"

    tr_raw = train_full[train_full["source_year"].isin(TRAIN_YEARS)].copy()
    va_raw = train_full[train_full["source_year"] == VAL_YEAR].copy()
    logger.info("Fold split: train(2023-24)=%s | val(2025)=%s",
                f"{len(tr_raw):,}", f"{len(va_raw):,}")

    # ----- Pass 1: fit aggregates on 2023-24, transform train + val -----
    # (the rows the model trains on, and the honest validation set)
    logger.info("Pass 1: fitting aggregates on 2023-24 fold -> transform train + val")
    tr_enc, va_enc, stats = _engineer_all(tr_raw.copy(), va_raw.copy(), cfg)

    # ----- Pass 2: fit aggregates on 2023-24 again, transform 2026 holdout -----
    # (same fit cohort -> identical aggregates; throwaway train copy discarded)
    logger.info("Pass 2: fitting aggregates on 2023-24 fold -> transform 2026 holdout")
    _, holdout, _ = _engineer_all(tr_raw.copy(), holdout.copy(), cfg)

    # ----- Reassemble full training frame so modeling.load_split() can split
    #       by source_year exactly as before — now fold-correct. -----
    train = pd.concat([tr_enc, va_enc], ignore_index=True)
    logger.info("Fold-scoped features: train_total=%s (2023-24 + 2025) | holdout=%s",
                f"{len(train):,}", f"{len(holdout):,}")

    # ----- Stats logging (from the 2023-24 fit, i.e. the cohort the model uses) -----
    eng_q_thresholds = stats["eng_q_thresholds"]
    logger.info(
        "  Engineer quartile thresholds: Q1<=%.1fd, Q2<=%.1fd, Q3<=%.1fd",
        eng_q_thresholds["q1"], eng_q_thresholds["q2"], eng_q_thresholds["q3"],
    )
    logger.info("  Segment delivery records: %d segments", len(stats["seg_delivery"]))
    logger.info("  Peak months: %s", sorted(stats["peak_months"]))
    logger.info("  Cities: %s | States: %s | global mean: %.3f",
                f"{stats['enc_diag']['n_cities']:,}", stats["enc_diag"]["n_states"],
                stats["enc_diag"]["global_mean"])

    # ----- 6J: feature spec -----
    logger.info("6J: Final feature spec")
    missing_feats = [f for f in MODEL_FEATURES if f not in train.columns]
    if missing_feats:
        logger.warning("Missing features: %s", missing_feats)
    else:
        logger.info("  All %d model features present", len(MODEL_FEATURES))

    # ----- 6K: missingness summary -----
    logger.info("6K: Missingness summary")
    missingness = build_missingness_summary(train, MODEL_FEATURES)
    missingness.to_csv(cfg.feature_dir / "missingness_summary.csv", index=False)
    n_fully = int((missingness["pct_missing"] == 0).sum())
    logger.info("  %d features fully populated (0%% missing)", n_fully)

    # ----- 6L: leakage review -----
    logger.info("6L: Leakage review")
    leakage_review = build_leakage_review()
    leakage_review.to_csv(cfg.feature_dir / "leakage_review.csv", index=False)

    # ----- 6M: data dictionary -----
    logger.info("6M: Data dictionary")
    data_dict = build_data_dictionary()
    data_dict.to_csv(cfg.feature_dir / "data_dictionary.csv", index=False)

    # ----- 6N: save feature tables -----
    logger.info("6N: Saving feature tables")
    save_cols = build_save_columns(train)
    train_fe = sanitize_for_parquet(train[save_cols])
    holdout_fe = sanitize_for_parquet(
        holdout[[c for c in save_cols if c in holdout.columns]]
    )
    train_fe.to_parquet(cfg.feature_dir / "feature_train.parquet", index=False)
    holdout_fe.to_parquet(cfg.feature_dir / "feature_holdout.parquet", index=False)

    feature_spec_tbl = build_feature_spec_table(train)
    feature_spec_tbl.to_csv(cfg.feature_dir / "feature_spec.csv", index=False)

    logger.info("  Train FE: %s | Holdout FE: %s",
                train_fe.shape, holdout_fe.shape)

    # ----- 6O: summary -----
    logger.info("=== Step 6 complete (fold-scoped) — %d model features ===",
                len(MODEL_FEATURES))

    return {
        "train_fe": train_fe,
        "holdout_fe": holdout_fe,
        "model_features": MODEL_FEATURES,
        "tier_stats": stats["tier_stats"],
        "channel_stats": stats["channel_stats"],
        "div_stats": stats["div_stats"],
        "seg_delivery": stats["seg_delivery"],
        "peak_months": stats["peak_months"],
        "engineer_quartile_thresholds": eng_q_thresholds,
        "feature_spec": feature_spec_tbl,
        "leakage_review": leakage_review,
        "data_dictionary": data_dict,
        "missingness": missingness,
    }


def run_feature_engineering_final(cfg: "FeatureConfig | None" = None) -> dict:
    """Build the Phase-2 'final' feature tables (aggregates fit on 2023-2025).

    Writes:
        feature_train_final.parquet    — all 2023-2025 rows, 2023-2025-fit
        feature_holdout_final.parquet  — 2026 holdout, 2023-2025-fit

    These feed modeling.run_final_holdout(), which retrains the locked
    models on the full 2023-2025 frame and scores 2026 once.
    """
    cfg = cfg or FeatureConfig()
    cfg.feature_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== PHASE 2: FINAL feature build (fit on all 2023-2025) ===")
    train_full = pd.read_parquet(cfg.interim_dir / "master_train.parquet")  # 2023-2025
    holdout = pd.read_parquet(cfg.interim_dir / "master_holdout.parquet")  # 2026
    assert holdout["source_year"].eq(2026).all(), "holdout must be 2026 only"
    logger.info("Train_full(2023-2025): %s | Holdout(2026): %s",
                f"{len(train_full):,}", f"{len(holdout):,}")

    # ---- ONE pass: fit every aggregate on the FULL 2023-2025 train,
    #      transform train_full + 2026 holdout. 2026 is strictly after the
    #      fit window, so this is leak-free. ----
    train_final, holdout_final, stats = _engineer_all(
        train_full.copy(), holdout.copy(), cfg,
    )

    eng_q = stats["eng_q_thresholds"]
    logger.info("  Engineer quartiles (2023-2025 fit): Q1<=%.1f Q2<=%.1f Q3<=%.1f",
                eng_q["q1"], eng_q["q2"], eng_q["q3"])
    logger.info("  Cities: %s | States: %s | global mean: %.3f",
                f"{stats['enc_diag']['n_cities']:,}", stats["enc_diag"]["n_states"],
                stats["enc_diag"]["global_mean"])

    # ---- save (mirror run_feature_engineering's 6N writer) ----
    save_cols = build_save_columns(train_final)
    train_fe = sanitize_for_parquet(train_final[save_cols])
    holdout_fe = sanitize_for_parquet(
        holdout_final[[c for c in save_cols if c in holdout_final.columns]]
    )
    train_fe.to_parquet(cfg.feature_dir / "feature_train_final.parquet", index=False)
    holdout_fe.to_parquet(cfg.feature_dir / "feature_holdout_final.parquet", index=False)
    logger.info("  Train FINAL: %s | Holdout FINAL: %s",
                train_fe.shape, holdout_fe.shape)
    logger.info("=== Phase 2 feature build complete ===")

    return {"train_final": train_fe, "holdout_final": holdout_fe, "stats": stats}

def _configure_logging() -> None:
    """Configure root logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    results = run_feature_engineering()
    logger.info("Feature engineering complete. %d model features.",
                len(results["model_features"]))

    final_results = run_feature_engineering_final()
    logger.info("Phase 2 final feature build complete. Train %s | Holdout %s",
                final_results["train_final"].shape,
                final_results["holdout_final"].shape)
