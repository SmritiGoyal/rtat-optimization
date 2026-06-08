"""
leakage_audit.py
================
Step 9: Final pre-submission leakage validation of all model features.

A standalone diagnostic module that runs five tests on every feature
used by the modeling pipeline. The audit's purpose is to catch subtle
target leakage *before* a model ships — the kind that produces
optimistic validation scores but collapses on real out-of-time data.

Five tests:

    Test 1: Manual timing classification
        Ground truth for each feature: was it computed from INTAKE,
        TRAINING aggregates, POST_INTAKE events, POST_CLOSE events,
        MEDIUM-risk (timing uncertain), or CONDITIONAL? The latter
        three categories are flagged for human review.

    Test 2: Target correlation
        Pearson correlation between feature and target_days. A single
        feature with |r| > 0.70 is suspicious; the typical max for
        clean tabular features in this domain is ~0.40.
        Threshold tiers: |r| > 0.80 HIGH, |r| > 0.70 ELEVATED, > 0.50 NOTE.

    Test 3: Train → holdout distribution stability
        Kolmogorov-Smirnov statistic comparing the feature distribution
        in train (2023-2025) vs holdout (2026). KS > 0.30 is flagged
        for investigation; KS > 0.15 is a moderate-shift note.

    Test 4: reclaim_period_days targeted check
        Specific to a feature family known to cause leakage when
        dominant. Verifies (a) low correlation with target_days and
        (b) low LightGBM importance.

    Test 5: Engineer-proxy future-data check
        Verifies ``engineer_hist_mean_rtat`` was computed only from
        training years. Reports the train-vs-holdout mean gap; gap
        < 1.0 day = stable / safe.

Verdict aggregation:
    The audit produces a per-feature verdict (CLEAN / REVIEW /
    CONFIRM-WITH-OPS / FLAG) and writes a CSV summary.

Pass criteria for the full audit:
    - Zero features in POST_CLOSE timing class (the two POST_CLOSE parts
      features, parts_delivery_tier and parts_has_arrival_flag, are excluded
      from MODEL_FEATURES and therefore absent from the feature parquet;
      they do not appear in audit results)
    - Zero features with |Pearson r| > 0.70
    - Zero features with KS > 0.30
    - reclaim_period_days correlation < 0.15, importance < 5%
    - Engineer proxy train-holdout gap < 1.0 day

Output artifact under ``outputs/models/``:
    leakage_audit.csv
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for the leakage audit stage."""

    # Directories
    interim_dir: Path = PROJECT_ROOT / "outputs" / "interim"
    feature_dir: Path = PROJECT_ROOT / "outputs" / "features"
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"

    # Test 2: correlation flag thresholds — preserved from notebook
    correlation_high: float = 0.80
    correlation_elevated: float = 0.70
    correlation_note: float = 0.50

    # Test 3: KS stability flag thresholds
    ks_large_shift: float = 0.30
    ks_moderate_shift: float = 0.15

    # Test 4: reclaim_period_days specific check thresholds
    reclaim_corr_max: float = 0.15
    reclaim_importance_max_pct: float = 5.0

    # Test 5: engineer proxy gap tolerance
    engineer_gap_max_days: float = 1.0

    # Minimum samples for statistical tests
    min_corr_samples: int = 1000
    min_train_samples_for_ks: int = 100
    min_holdout_samples_for_ks: int = 10


# =====================================================================
# TIMING_AUDIT — manual ground-truth classification per feature
# =====================================================================
# Maintained as a literal dictionary. A new feature added to the model
# without an entry here will default to UNKNOWN, surfacing in the report
# as ⚠ REVIEW. This is a deliberate fail-safe.

TIMING_AUDIT: dict[str, tuple[str, str]] = {
    # --- Geography ---
    "market_tier_ordinal": (
        "INTAKE", "Market category known at repair open",
    ),
    "tier_mean_rtat": (
        "TRAINING", "Mean RTAT per tier — training aggregate only",
    ),
    "tier_late_rate5": (
        "TRAINING", "Late rate per tier — training aggregate only",
    ),
    "city_target_enc": (
        "TRAINING", "Smoothed city mean — training aggregate, k=30",
    ),
    "state_target_enc": (
        "TRAINING", "Smoothed state mean — training aggregate, k=30",
    ),
    # --- Channel ---
    "channel_risk_ordinal": (
        "INTAKE", "Channel known at repair open",
    ),
    "channel_mean_rtat": (
        "TRAINING", "Mean RTAT per channel — training aggregate",
    ),
    "channel_late_rate5": (
        "TRAINING", "Late rate per channel — training aggregate",
    ),
    # --- Time ---
    "month_of_year": (
        "INTAKE", "Month derived from close date — known at close",
    ),
    "quarter": (
        "INTAKE", "Quarter — known at close",
    ),
    "day_of_week": (
        "INTAKE", "Day of week — known at close",
    ),
    "is_weekend_close": (
        "INTAKE", "Weekend flag — known at close",
    ),
    "is_peak_month": (
        "TRAINING", "Peak month flag — based on training late rates",
    ),
    "month_mean_rtat": (
        "TRAINING", "Monthly mean — training aggregate",
    ),
    # --- Product ---
    "div_mean_rtat": (
        "TRAINING", "Division mean — training aggregate",
    ),
    "div_late_rate5": (
        "TRAINING", "Division late rate — training aggregate",
    ),
    "is_ter_repair": (
        "MEDIUM", "TER flag — timing of assignment needs operations confirmation",
    ),
    "is_sealed_repair": (
        "MEDIUM", "Sealed flag — timing of assignment needs operations confirmation",
    ),
    # --- Engineer ---
    "engineer_hist_mean_rtat": (
        "TRAINING", "Historical mean from prior repairs — training years only",
    ),
    "engineer_quartile": (
        "TRAINING", "Derived from engineer_hist_mean_rtat",
    ),
    "engineer_proxy_missing": (
        "TRAINING", "Admin flag for missing proxy",
    ),
    # --- Reclaim ---
    "has_parts_reclaim": (
        "INTAKE", "Parts presence at intake — Reclaim Parts_No1-5",
    ),
    "parts_count_reclaim": (
        "INTAKE", "Parts count at intake",
    ),
    "parts_complexity_score": (
        "INTAKE", "Composite of intake-known parts signals",
    ),
    "is_reclaim": (
        "INTAKE", "Prior visit history — known at intake",
    ),
    "is_same_symptom_reclaim": (
        "INTAKE", "Symptom comparison to prior visit — known at intake",
    ),
    "reclaim_period_days": (
        "INTAKE", "Days since prior visit — historical, known at intake",
    ),
    # --- DMS / Parts logistics ---
    "ordered_via_dms": (
        "POST_INTAKE", "Parts order placed after intake, before close",
    ),
    "parts_line_count": (
        "POST_INTAKE", "Part lines ordered — after intake",
    ),
    "parts_order_qty_sum": (
        "POST_INTAKE", "Parts quantity — after intake",
    ),
    "parts_multi_line_flag": (
        "POST_INTAKE", "Multi-line flag — after intake",
    ),
    "parts_has_arrival_flag": (
        "POST_CLOSE", "Arrival flag — may be after close. CHECK.",
    ),
    "parts_has_shipment_flag": (
        "POST_INTAKE", "Shipment flag — between intake and close",
    ),
    "parts_shipping_tier": (
        "POST_INTAKE", "Shipping method — known when ordered",
    ),
    "parts_delivery_tier": (
        "POST_CLOSE", "CONDITIONAL — requires arrival to have occurred",
    ),
    "parts_delivery_tier_known": (
        "POST_INTAKE", "Flag only — no timing issue",
    ),
    "seg_delivery_days_hist": (
        "TRAINING", "Segment-level median — training aggregate. SAFE.",
    ),
    "parts_truncation_flag": (
        "TRAINING", "Data quality flag — no timing issue",
    ),
    # --- Interactions ---
    "geo_channel_risk": (
        "INTAKE", "Product of two intake-known features",
    ),
    "rural_parts_flag": (
        "INTAKE", "Product of two intake-known features",
    ),
    "eng_channel_risk": (
        "MEDIUM", "Engineer × channel — engineer timing needs confirmation",
    ),
}


# Default feature list — kept in sync with feature_engineering.MODEL_FEATURES
# (and modeling.EXTENDED_FEATURES adjusted to drop parts_order_to_arrival_days_safe
# and include is_weekend_close + day_of_week + seg_delivery_days_hist as in
# the original Step 9 cell).

EXTENDED_FEATURES: list[str] = [
    "market_tier_ordinal", "tier_mean_rtat", "tier_late_rate5",
    "city_target_enc", "state_target_enc",
    "channel_risk_ordinal", "channel_mean_rtat", "channel_late_rate5",
    "month_of_year", "quarter", "is_peak_month", "month_mean_rtat",
    "div_mean_rtat", "div_late_rate5", "is_ter_repair",
    "engineer_hist_mean_rtat", "engineer_quartile", "engineer_proxy_missing",
    "has_parts_reclaim", "parts_count_reclaim", "parts_complexity_score",
    "is_reclaim", "is_same_symptom_reclaim", "reclaim_period_days",
    "ordered_via_dms", "parts_line_count", "parts_order_qty_sum",
    "parts_multi_line_flag", "parts_has_arrival_flag", "parts_has_shipment_flag",
    "parts_shipping_tier", "parts_delivery_tier", "parts_delivery_tier_known",
    "seg_delivery_days_hist", "parts_truncation_flag",
    "geo_channel_risk", "rural_parts_flag", "eng_channel_risk",
    "is_sealed_repair", "is_weekend_close", "day_of_week",
]


# =====================================================================
# DATACLASSES — structured audit outputs
# =====================================================================

@dataclass
class AuditResults:
    """Aggregated results across all audit tests."""
    timing_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    reclaim_check: dict = field(default_factory=dict)
    engineer_check: dict = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict_counts: dict[str, int] = field(default_factory=dict)
    features_flagged: list[str] = field(default_factory=list)


# =====================================================================
# DATA LOADING
# =====================================================================

def load_audit_inputs(cfg: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load feature parquets, attach target_days + source_year from master.

    Returns:
        train         : train feature parquet enriched with target_days
                        and source_year from interim master
        holdout       : holdout feature parquet
        feature_importance : dict feature → gain % from primary LightGBM
                             classifier (T=5). Used by Test 4.
    """
    train = pd.read_parquet(cfg.feature_dir / "feature_train.parquet")
    holdout = pd.read_parquet(cfg.feature_dir / "feature_holdout.parquet")

    master_train_path = cfg.interim_dir / "master_train.parquet"
    if master_train_path.exists():
        master_cols = pd.read_parquet(
            master_train_path,
            columns=["repair_no_clean", "Warranty_Closed_Date",
                     "source_year", "target_days"],
        )
        add = [c for c in ("source_year", "target_days", "Warranty_Closed_Date")
               if c not in train.columns]
        if add:
            train = train.merge(
                master_cols[["repair_no_clean", *add]],
                on="repair_no_clean", how="left",
            )

    # Optional: feature importance from primary classifier
    importance: dict[str, float] = {}
    primary_model_path = cfg.model_dir / "lgbm_ontime5.pkl"
    if primary_model_path.exists():
        try:
            import joblib
            model = joblib.load(primary_model_path)
            gains = model.booster_.feature_importance(importance_type="gain")
            names = model.booster_.feature_name()
            total = float(gains.sum()) if gains.sum() > 0 else 1.0
            importance = {n: 100.0 * g / total for n, g in zip(names, gains)}
        except Exception as e:
            logger.warning("Could not load feature importance: %s", e)

    return train, holdout, importance


# =====================================================================
# TEST 1 — TIMING CLASSIFICATION
# =====================================================================

def run_test1_timing(model_features: list[str]) -> pd.DataFrame:
    """Manual ground-truth lookup for each feature's computation source.

    Features not in TIMING_AUDIT default to ``UNKNOWN`` — deliberate
    fail-safe so a new feature added without an audit entry surfaces
    as a flag.
    """
    rows = []
    for feat in model_features:
        timing, note = TIMING_AUDIT.get(feat, ("UNKNOWN", "Not in audit table"))
        rows.append({
            "feature": feat,
            "timing": timing,
            "note": note,
            "in_model": True,
        })
    return pd.DataFrame(rows).sort_values("timing").reset_index(drop=True)


# =====================================================================
# TEST 2 — TARGET CORRELATION
# =====================================================================

def _corr_flag(r: float, cfg: AuditConfig) -> str:
    """Map a Pearson r value to a flag tier (HIGH / ELEVATED / NOTE / empty)."""
    if np.isnan(r):
        return ""
    abs_r = abs(r)
    if abs_r > cfg.correlation_high:
        return "⚠ HIGH — investigate"
    if abs_r > cfg.correlation_elevated:
        return "⚠ ELEVATED — review"
    if abs_r > cfg.correlation_note:
        return "NOTE — moderate"
    return ""


def run_test2_correlation(
    train: pd.DataFrame,
    model_features: list[str],
    cfg: AuditConfig,
) -> pd.DataFrame:
    """Pearson correlation between each feature and target_days.

    Skips features with fewer than ``cfg.min_corr_samples`` non-null rows.
    Categorical / non-numeric columns get NaN correlation. Results sorted
    by absolute correlation descending.
    """
    if "target_days" not in train.columns:
        logger.warning("target_days missing — skipping Test 2")
        return pd.DataFrame(columns=["feature", "pearson_r", "n_valid", "flag"])

    train_cohort = train[train["target_days"].notna()].copy()

    rows = []
    for feat in model_features:
        if feat not in train_cohort.columns:
            continue
        col = train_cohort[feat]
        if str(col.dtype).startswith("Int"):
            col = col.astype("float")

        non_null = col.notna()
        if non_null.sum() < cfg.min_corr_samples:
            rows.append({
                "feature": feat, "pearson_r": np.nan,
                "n_valid": int(non_null.sum()), "flag": "",
            })
            continue

        if col.dtype == object:
            rows.append({
                "feature": feat, "pearson_r": np.nan,
                "n_valid": int(non_null.sum()),
                "flag": "categorical — skipped",
            })
            continue

        try:
            r, _ = stats.pearsonr(
                col[non_null].astype(float),
                train_cohort.loc[non_null, "target_days"].astype(float),
            )
        except Exception:
            r = np.nan

        rows.append({
            "feature": feat,
            "pearson_r": round(float(r), 4) if not np.isnan(r) else np.nan,
            "n_valid": int(non_null.sum()),
            "flag": _corr_flag(r, cfg),
        })

    return pd.DataFrame(rows).sort_values(
        "pearson_r", key=lambda s: s.abs(), ascending=False, na_position="last",
    ).reset_index(drop=True)


# =====================================================================
# TEST 3 — TRAIN → HOLDOUT STABILITY
# =====================================================================

def _ks_flag(ks: float, cfg: AuditConfig) -> str:
    """Map a KS statistic to a flag tier (LARGE / MODERATE / empty)."""
    if np.isnan(ks):
        return ""
    if ks > cfg.ks_large_shift:
        return "⚠ LARGE SHIFT — investigate"
    if ks > cfg.ks_moderate_shift:
        return "NOTE — moderate shift"
    return ""


def _stability_row(
    train_col: pd.Series,
    holdout_col: pd.Series,
    feature: str,
    cfg: AuditConfig,
) -> dict:
    """Compute one row of the stability table for a single feature."""
    base = {
        "feature": feature, "ks_stat": np.nan,
        "tr_mean": np.nan, "ho_mean": np.nan,
        "mean_shift_pct": np.nan, "flag": "",
    }

    if str(train_col.dtype).startswith("Int"):
        train_col = train_col.astype("float")
    if str(holdout_col.dtype).startswith("Int"):
        holdout_col = holdout_col.astype("float")

    if train_col.dtype == object or holdout_col.dtype == object:
        base["tr_mean"] = "categorical"
        base["ho_mean"] = "categorical"
        base["flag"] = "categorical — use value counts instead"
        return base

    try:
        tr_vals = train_col.dropna().astype(float).values
        ho_vals = holdout_col.dropna().astype(float).values
    except (ValueError, TypeError):
        base["tr_mean"] = "non-numeric"
        base["ho_mean"] = "non-numeric"
        base["flag"] = "non-numeric — skipped"
        return base

    if len(tr_vals) < cfg.min_train_samples_for_ks:
        base["flag"] = "insufficient train data"
        return base
    if len(ho_vals) < cfg.min_holdout_samples_for_ks:
        base["flag"] = "insufficient holdout data"
        return base

    ks_stat, _ = stats.ks_2samp(tr_vals, ho_vals)
    tr_mean = float(tr_vals.mean())
    ho_mean = float(ho_vals.mean())
    mean_shift = abs(ho_mean - tr_mean) / (abs(tr_mean) + 1e-9)

    return {
        "feature": feature,
        "ks_stat": round(float(ks_stat), 4),
        "tr_mean": round(tr_mean, 4),
        "ho_mean": round(ho_mean, 4),
        "mean_shift_pct": round(float(mean_shift) * 100, 2),
        "flag": _ks_flag(float(ks_stat), cfg),
    }


def run_test3_stability(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    model_features: list[str],
    cfg: AuditConfig,
) -> pd.DataFrame:
    """KS two-sample test per feature comparing train vs holdout.

    A large KS statistic (>0.30) flags either legitimate temporal drift
    (acceptable) or feature leakage (concerning). The audit surfaces the
    signal — a human interprets it.
    """
    rows: list[dict] = []
    for feat in model_features:
        if feat not in train.columns or feat not in holdout.columns:
            continue
        rows.append(_stability_row(train[feat], holdout[feat], feat, cfg))
    return pd.DataFrame(rows).sort_values(
        "ks_stat", ascending=False, na_position="last",
    ).reset_index(drop=True)


# =====================================================================
# TEST 4 — RECLAIM_PERIOD_DAYS TARGETED CHECK
# =====================================================================

def run_test4_reclaim_period(
    train: pd.DataFrame,
    feature_importance: dict[str, float],
    cfg: AuditConfig,
) -> dict:
    """Targeted timing check for ``reclaim_period_days``.

    Two conditions for pass:
        (a) Correlation with target_days < ``cfg.reclaim_corr_max``
        (b) LightGBM importance < ``cfg.reclaim_importance_max_pct``

    The feature measures the gap between a prior repair and the current
    one — semantically pre-close information. But because it's null for
    non-reclaim cases (~92% of rows) and otherwise temporally proximate
    to the target, it has been a leakage vector in adjacent analyses.
    This test verifies both conditions empirically.
    """
    result = {
        "feature": "reclaim_period_days",
        "present": False,
        "verdict": "N/A — feature not in pipeline",
    }

    if ("reclaim_period_days" not in train.columns
            or "target_days" not in train.columns):
        return result

    sub = train[["reclaim_period_days", "target_days"]].dropna()
    if sub.empty:
        result["verdict"] = "N/A — no non-null values"
        return result

    corr = float(sub["reclaim_period_days"].corr(sub["target_days"]))
    importance = float(feature_importance.get("reclaim_period_days", 0.0))

    result.update({
        "present": True,
        "n_nonnull": int(len(sub)),
        "mean_days": round(float(sub["reclaim_period_days"].mean()), 2),
        "median_days": round(float(sub["reclaim_period_days"].median()), 2),
        "max_days": round(float(sub["reclaim_period_days"].max()), 2),
        "corr_with_target": round(corr, 4),
        "importance_pct": round(importance, 2),
    })

    corr_ok = abs(corr) < cfg.reclaim_corr_max
    imp_ok = importance < cfg.reclaim_importance_max_pct
    if corr_ok and imp_ok:
        result["verdict"] = "✓ PASS — low correlation and low importance"
    elif not corr_ok:
        result["verdict"] = f"⚠ FAIL — correlation {corr:+.3f} exceeds threshold"
    else:
        result["verdict"] = f"⚠ FAIL — importance {importance:.1f}% exceeds threshold"

    return result


# =====================================================================
# TEST 5 — ENGINEER PROXY FUTURE-DATA CHECK
# =====================================================================

def run_test5_engineer_proxy(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    cfg: AuditConfig,
) -> dict:
    """Verify ``engineer_hist_mean_rtat`` train-vs-holdout consistency.

    The historical mean should be computed from training years only.
    A small train-vs-holdout gap (< 1 day) indicates the encoder was
    correctly scoped. A large gap could indicate either legitimate
    drift (acceptable) or leakage where the encoder accidentally saw
    holdout data.

    Also reports the per-year breakdown within training so we can
    visualize the source-year stability inside the training window.
    """
    result = {
        "feature": "engineer_hist_mean_rtat",
        "present": False,
        "verdict": "N/A — feature not in pipeline",
    }
    if "engineer_hist_mean_rtat" not in train.columns:
        return result

    train_mean = float(train["engineer_hist_mean_rtat"].mean())
    holdout_mean = (
        float(holdout["engineer_hist_mean_rtat"].mean())
        if "engineer_hist_mean_rtat" in holdout.columns
        else float("nan")
    )

    result["present"] = True
    result["train_mean"] = round(train_mean, 3)
    result["holdout_mean"] = round(holdout_mean, 3) if not np.isnan(holdout_mean) else np.nan

    # Per-year breakdown if source_year is available
    if "source_year" in train.columns:
        by_year = (train
                   .groupby("source_year")["engineer_hist_mean_rtat"]
                   .agg(["mean", "std", "count"])
                   .round(3))
        result["train_by_year"] = by_year.to_dict()

    gap = abs(holdout_mean - train_mean) if not np.isnan(holdout_mean) else 0.0
    result["train_holdout_gap_days"] = round(float(gap), 3)

    if np.isnan(holdout_mean):
        result["verdict"] = "⚠ WARNING — holdout data missing"
    elif gap < cfg.engineer_gap_max_days:
        result["verdict"] = f"✓ PASS — gap {gap:.2f}d within tolerance"
    else:
        result["verdict"] = f"NOTE — gap {gap:.2f}d > tolerance (acceptable drift?)"

    return result


# =====================================================================
# VERDICT AGGREGATION
# =====================================================================

def _overall_verdict(row: pd.Series) -> str:
    """Determine the overall per-feature verdict from the three tests.

    Priority order (preserved from notebook):
        1. timing == POST_CLOSE or UNKNOWN  → ⚠ REVIEW
        2. timing == MEDIUM                  → ⚠ CONFIRM WITH OPS
        3. corr_flag contains ⚠              → ⚠ CORRELATION FLAG
        4. stab_flag contains ⚠              → ⚠ STABILITY FLAG
        5. else                              → ✓ CLEAN
    """
    timing = row.get("timing")
    if timing in ("POST_CLOSE", "UNKNOWN"):
        return "⚠ REVIEW"
    if timing == "MEDIUM":
        return "⚠ CONFIRM WITH OPS"
    if "⚠" in str(row.get("corr_flag", "")):
        return "⚠ CORRELATION FLAG"
    if "⚠" in str(row.get("stab_flag", "")):
        return "⚠ STABILITY FLAG"
    return "✓ CLEAN"


def aggregate_verdicts(
    timing_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    stab_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the per-test tables into a single verdict table.

    The output has one row per feature with columns from all three
    tests plus an ``overall_verdict`` that follows the priority order
    defined in ``_overall_verdict``.
    """
    merged = timing_df[["feature", "timing", "note"]].copy()
    merged = merged.merge(
        corr_df[["feature", "pearson_r", "flag"]].rename(
            columns={"flag": "corr_flag"},
        ),
        on="feature", how="left",
    )
    merged = merged.merge(
        stab_df[["feature", "ks_stat", "flag"]].rename(
            columns={"flag": "stab_flag"},
        ),
        on="feature", how="left",
    )
    merged["overall_verdict"] = merged.apply(_overall_verdict, axis=1)
    return merged


# =====================================================================
# MAIN ORCHESTRATION
# =====================================================================

def run_audit(
    cfg: AuditConfig | None = None,
    model_features: list[str] = EXTENDED_FEATURES,
) -> AuditResults:
    """Run the full 5-test audit and write CSV summary.

    Args:
        cfg: Audit configuration (defaults to AuditConfig()).
        model_features: Features to audit. Defaults to the production
            EXTENDED feature set; pass a subset to audit a candidate model.

    Returns:
        Populated ``AuditResults`` with per-feature verdicts, the two
        targeted-test summaries, verdict counts, and the list of
        features that received any non-✓ verdict.
    """
    cfg = cfg or AuditConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    train, holdout, importance = load_audit_inputs(cfg)
    # Trim model_features to those present in the feature parquet
    model_features = [f for f in model_features if f in train.columns]
    logger.info(
        "Auditing %d features | train: %s | holdout: %s",
        len(model_features), f"{len(train):,}", f"{len(holdout):,}",
    )

    # ---- Test 1: timing ----
    logger.info("=== Test 1: Timing classification ===")
    timing_df = run_test1_timing(model_features)
    for timing_class, count in timing_df["timing"].value_counts().items():
        logger.info("  %-12s : %d features", timing_class, count)
    attention_count = int(timing_df["timing"].isin(
        ["MEDIUM", "POST_CLOSE", "UNKNOWN", "CONDITIONAL"],
    ).sum())
    if attention_count:
        logger.info("  %d feature(s) need human review on timing", attention_count)

    # ---- Test 2: target correlation ----
    logger.info("=== Test 2: Target correlation ===")
    corr_df = run_test2_correlation(train, model_features, cfg)
    flagged_corr = corr_df[
        corr_df["flag"].astype(str).str.contains("⚠", na=False)
    ]
    logger.info("  %d feature(s) with |r| > %.2f",
                len(flagged_corr), cfg.correlation_elevated)

    # ---- Test 3: stability ----
    logger.info("=== Test 3: Train→holdout stability ===")
    stab_df = run_test3_stability(train, holdout, model_features, cfg)
    flagged_stab = stab_df[
        stab_df["flag"].astype(str).str.contains("⚠", na=False)
    ]
    logger.info("  %d feature(s) with KS > %.2f",
                len(flagged_stab), cfg.ks_large_shift)

    # ---- Test 4: reclaim_period_days ----
    logger.info("=== Test 4: reclaim_period_days targeted check ===")
    test4 = run_test4_reclaim_period(train, importance, cfg)
    logger.info("  %s", test4.get("verdict"))

    # ---- Test 5: engineer proxy ----
    logger.info("=== Test 5: Engineer proxy future-data check ===")
    test5 = run_test5_engineer_proxy(train, holdout, cfg)
    logger.info("  %s", test5.get("verdict"))

    # ---- Aggregate verdicts ----
    summary = aggregate_verdicts(timing_df, corr_df, stab_df)
    summary.to_csv(cfg.model_dir / "leakage_audit.csv", index=False)

    counts = summary["overall_verdict"].value_counts().to_dict()
    flagged = summary[
        summary["overall_verdict"].str.contains("⚠", na=False)
    ]["feature"].tolist()

    logger.info("=== AUDIT SUMMARY ===")
    for verdict, count in counts.items():
        logger.info("  %-30s : %d", verdict, count)
    n_clean = counts.get("✓ CLEAN", 0)
    logger.info("  %d of %d features fully validated (✓ CLEAN)",
                n_clean, len(summary))
    if flagged:
        logger.info("  Features requiring review: %s",
                    ", ".join(flagged[:5]) + ("..." if len(flagged) > 5 else ""))

    return AuditResults(
        timing_df=timing_df,
        correlation_df=corr_df,
        stability_df=stab_df,
        reclaim_check=test4,
        engineer_check=test5,
        summary=summary,
        verdict_counts=counts,
        features_flagged=flagged,
    )


def _configure_logging() -> None:
    """Configure root logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    results = run_audit()
    logger.info(
        "Audit complete. %d features audited, %d flagged for review.",
        len(results.summary), len(results.features_flagged),
    )
