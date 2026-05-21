"""
prioritization.py
=================
Step 8: Segment prioritization and resource allocation output.

Translates model predictions into the project's core business
deliverable: which Market_Category × Channel segments should receive
resources first, under which target threshold (T = 3, 5, 7, 10 days),
and through which operational lever.

Business questions answered:
    Q1: For target T, which segments get resources first?
    Q2: Is the main delay driver parts, engineer, channel, or complexity?
    Q3: Which lever produces the most improvement per segment?

Method:
    1. Generate predicted delay risk for all 2023-2025 repairs
    2. Build segment priority matrix: volume × late rate × lever
    3. Rank segments by combined impact score per threshold
    4. Decompose delay drivers per segment using feature values
       against global benchmarks — 4 levers, primary + secondary
    5. Show how priority shifts across T = 3, 5, 7, 10
    6. NPS validation layer (2025 NPS responder subset)
    7. Produce final recommendation table for presentation

Output artifacts under ``outputs/prioritization/``:
    priority_matrix.csv
    lever_decomposition.csv
    threshold_shift.csv
    tier_summary.csv
    nps_validation.csv
    nps_by_tier.csv
    final_recommendation.csv
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb  # noqa: F401  (only needed if running standalone)
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PrioritizationConfig:
    """Configuration for the prioritization stage."""

    # Directories
    interim_dir: Path = PROJECT_ROOT / "outputs" / "interim"
    feature_dir: Path = PROJECT_ROOT / "outputs" / "features"
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"
    output_dir: Path = PROJECT_ROOT / "outputs" / "prioritization"

    # Thresholds to score and rank against
    ontime_targets: tuple[int, ...] = (3, 5, 7, 10)

    # Minimum segment size for inclusion in the priority matrix
    min_segment_repairs: int = 500

    # Number of top segments retained in the final recommendation table
    top_n_segments: int = 10

    # Lever scoring multipliers — preserved from original notebook
    parts_rate_mult: float = 1.15
    parts_delivery_mult: float = 1.30
    engineer_q4_strong_mult: float = 1.30
    engineer_q4_weak_mult: float = 1.10
    sealed_rate_mult: float = 1.30
    reclaim_rate_mult: float = 1.30

    # Tier ordering for elevation-based improvement estimate
    tier_order: tuple[str, ...] = (
        "1. Top 10", "2. Metro", "3. Urban", "4. Rural",
    )

    # NPS pred_risk bucket edges (5 bins: very low → very high)
    nps_risk_bins: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    nps_risk_labels: tuple[str, ...] = (
        "Very low\n(0-20%)", "Low\n(20-40%)",
        "Medium\n(40-60%)", "High\n(60-80%)",
        "Very high\n(80-100%)",
    )

    # Year used for NPS post-hoc validation (best coverage in 2025)
    nps_validation_year: int = 2025


# Channel ordinal — preserved exactly from notebook for lever scoring
CHANNEL_MAP: dict[str, int] = {
    "DMS":             1,
    "DMS2":            2,
    "ASC":             3,
    "Premier Partner": 4,
    "ASD":             5,
    "AE":              6,
    "SPO":             7,
}


# Extended features needed for prediction (must match modeling.py's
# EXTENDED_FEATURES). Kept here as a literal list so this module can
# run standalone without importing from modeling.py.
EXTENDED_FEATURES: list[str] = [
    "market_tier_ordinal", "tier_mean_rtat", "tier_late_rate5",
    "city_target_enc", "state_target_enc",
    "channel_risk_ordinal", "channel_mean_rtat", "channel_late_rate5",
    "month_of_year", "quarter", "is_peak_month", "month_mean_rtat",
    "div_mean_rtat", "div_late_rate5", "is_ter_repair",
    "engineer_hist_mean_rtat", "engineer_quartile",
    "has_parts_reclaim", "parts_count_reclaim", "parts_complexity_score",
    "is_reclaim", "is_same_symptom_reclaim",
    "ordered_via_dms", "parts_delivery_tier_known",
    "geo_channel_risk", "rural_parts_flag", "eng_channel_risk",
    "parts_line_count", "parts_order_qty_sum", "parts_multi_line_flag",
    "parts_has_arrival_flag", "parts_has_shipment_flag",
    "parts_delivery_tier", "parts_order_to_arrival_days_safe",
    "parts_truncation_flag", "reclaim_period_days", "is_sealed_repair",
]


# Master context columns to merge onto feature parquets for segment labels
MASTER_CONTEXT_COLS: list[str] = [
    "repair_no_clean", "Market_Category", "Channel",
    "Division_Name", "State_", "City_", "General_Market",
    "source_year", "OnTime_3", "OnTime_5", "OnTime_7",
    "OnTime_10", "target_days", "has_nps",
    "is_promoter", "is_detractor",
]


# =====================================================================
# SECTION 8A: LOAD MODELS + DATA
# =====================================================================

def load_models_and_data(cfg: PrioritizationConfig) -> dict[str, object]:
    """Load LightGBM classifiers/regressor + feature tables + master context.

    Returns a dict containing:
        lgb_models   : dict[T → LGBMClassifier] for T in cfg.ontime_targets
        lgb_reg      : fitted LGBMRegressor on target_days
        train_feat   : feature parquet for 2023-2025
        holdout_feat : feature parquet for 2026
        master_train : master context (Market_Category, Channel, NPS, etc.)
        master_hold  : master context for holdout
    """
    lgb_models: dict[int, object] = {}
    for t in cfg.ontime_targets:
        path = cfg.model_dir / f"lgbm_ontime{t}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing classification model: {path}")
        lgb_models[t] = joblib.load(path)

    lgb_reg = joblib.load(cfg.model_dir / "lgbm_regression.pkl")
    logger.info("Loaded %d LightGBM classifiers + regression model", len(lgb_models))

    train_feat = pd.read_parquet(cfg.feature_dir / "feature_train.parquet")
    holdout_feat = pd.read_parquet(cfg.feature_dir / "feature_holdout.parquet")

    master_train = pd.read_parquet(
        cfg.interim_dir / "master_train.parquet",
        columns=MASTER_CONTEXT_COLS,
    )
    master_hold = pd.read_parquet(
        cfg.interim_dir / "master_holdout.parquet",
        columns=MASTER_CONTEXT_COLS,
    )

    return {
        "lgb_models": lgb_models,
        "lgb_reg": lgb_reg,
        "train_feat": train_feat,
        "holdout_feat": holdout_feat,
        "master_train": master_train,
        "master_hold": master_hold,
    }


def merge_context(feat_df: pd.DataFrame, ctx_df: pd.DataFrame) -> pd.DataFrame:
    """Bring segment labels and NPS columns onto a feature parquet.

    The feature parquet contains modeling features only; the master
    parquet has the segment / NPS columns. This function joins them on
    repair_no_clean, skipping columns that already exist in feat_df.
    """
    ctx_cols = [
        "repair_no_clean", "Market_Category", "Channel",
        "Division_Name", "State_", "City_", "source_year",
        "has_nps", "is_promoter", "is_detractor",
    ]
    add = [c for c in ctx_cols if c not in feat_df.columns or c == "repair_no_clean"]
    return feat_df.merge(ctx_df[add], on="repair_no_clean", how="left")


# =====================================================================
# SECTION 8B: PREDICTION GENERATION
# =====================================================================

def get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the feature matrix, converting Int dtypes to float32."""
    X = df[EXTENDED_FEATURES].copy()
    for col in X.columns:
        if str(X[col].dtype).startswith("Int"):
            X[col] = X[col].astype("float32")
    return X


def attach_predictions(
    df: pd.DataFrame,
    lgb_models: dict[int, object],
    lgb_reg: object,
    thresholds: tuple[int, ...],
) -> pd.DataFrame:
    """Score per-threshold late probability and predicted RTAT.

    For each threshold T, attaches ``pred_late_prob_{T}`` =
    1 - P(OnTime_T). Also attaches a continuous ``pred_rtat`` from
    the regression model.
    """
    df = df.copy()
    X = get_X(df)
    for t in thresholds:
        df[f"pred_late_prob_{t}"] = 1 - lgb_models[t].predict_proba(X)[:, 1]
    df["pred_rtat"] = lgb_reg.predict(X)
    return df


# =====================================================================
# SECTION 8C: SEGMENT PRIORITY MATRIX
# =====================================================================

def build_priority_matrix(
    df: pd.DataFrame,
    cfg: PrioritizationConfig,
) -> pd.DataFrame:
    """Build the Market_Category × Channel segment priority matrix.

    For each segment with ≥ ``cfg.min_segment_repairs`` repairs, computes:
        - n_repairs and share of total
        - mean_rtat_actual and mean_rtat_pred
        - actual_late_{T} and pred_late_{T} for T in ontime_targets
        - cases_at_risk_{T} = n_repairs × actual_late_{T}
        - lever indicators: parts_rate, engineer_q4_rate, dms_rate,
          sealed_rate, reclaim_rate, delivery_days_median
        - priority_score_5 = n_repairs × actual_late_5
        - priority_rank_5

    Returns the matrix sorted by priority_score_5 descending.
    """
    rows: list[dict] = []
    n_total = len(df)

    for (tier, channel), g in df.groupby(["Market_Category", "Channel"]):
        if len(g) < cfg.min_segment_repairs:
            continue
        rows.append(_segment_row(g, tier, channel, n_total, cfg))

    df_out = pd.DataFrame(rows)

    df_out["priority_score_5"] = (
        df_out["n_repairs"] * df_out["actual_late_5"]
    ).round(0).astype(int)
    df_out["priority_rank_5"] = df_out["priority_score_5"].rank(
        ascending=False, method="min",
    ).astype(int)

    return df_out.sort_values("priority_score_5", ascending=False).reset_index(drop=True)


def _segment_row(
    g: pd.DataFrame,
    tier: str,
    channel: str,
    n_total: int,
    cfg: PrioritizationConfig,
) -> dict:
    """Compute one row of the priority matrix for a single segment."""
    row: dict = {
        "Market_Category": tier,
        "Channel": channel,
        "n_repairs": len(g),
        "pct_of_total": round(len(g) / max(n_total, 1), 4),
        "mean_rtat_actual": round(float(g["target_days"].mean()), 3),
        "mean_rtat_pred": round(float(g["pred_rtat"].mean()), 3),
    }

    for t in cfg.ontime_targets:
        actual_late = 1 - g[f"OnTime_{t}"].astype(float).mean()
        pred_late = g[f"pred_late_prob_{t}"].mean()
        row[f"actual_late_{t}"] = round(float(actual_late), 4)
        row[f"pred_late_{t}"] = round(float(pred_late), 4)
        row[f"cases_at_risk_{t}"] = int(len(g) * actual_late)

    # Lever indicators — median / mean per segment
    row["parts_rate"] = round(float(g["has_parts_reclaim"].astype(float).mean()), 4)
    row["engineer_q4_rate"] = round(float((g["engineer_quartile"] == 4).mean()), 4)
    row["dms_rate"] = round(float(g["ordered_via_dms"].astype(float).mean()), 4)
    row["sealed_rate"] = round(float(g["is_sealed_repair"].astype(float).mean()), 4)
    row["reclaim_rate"] = round(float(g["is_reclaim"].astype(float).mean()), 4)
    row["delivery_days_median"] = (
        round(float(g["parts_order_to_arrival_days_safe"].median()), 3)
        if g["parts_order_to_arrival_days_safe"].notna().sum() > 10
        else np.nan
    )
    return row


# =====================================================================
# SECTION 8D: LEVER DECOMPOSITION — 4 levers, primary + secondary
# =====================================================================

@dataclass
class GlobalBenchmarks:
    """Population-level reference values used to score segment levers."""
    parts_rate: float = 0.0
    engineer_q4_rate: float = 0.0
    parts_delivery_days: float = 0.0
    sealed_rate: float = 0.0
    reclaim_rate: float = 0.0
    mean_rtat: float = 0.0


def compute_global_benchmarks(df: pd.DataFrame) -> GlobalBenchmarks:
    """Compute population-level benchmarks for the lever decomposition.

    Returns the six benchmark statistics used as the comparison baseline
    in ``classify_primary_lever``: overall parts rate, Q4 engineer rate,
    median parts delivery days, sealed rate, reclaim rate, and mean RTAT.
    """
    return GlobalBenchmarks(
        parts_rate=float(df["has_parts_reclaim"].astype(float).mean()),
        engineer_q4_rate=float((df["engineer_quartile"] == 4).mean()),
        parts_delivery_days=float(df["parts_order_to_arrival_days_safe"].median()),
        sealed_rate=float(df["is_sealed_repair"].astype(float).mean()),
        reclaim_rate=float(df["is_reclaim"].astype(float).mean()),
        mean_rtat=float(df["target_days"].mean()),
    )


def classify_primary_lever(
    row: pd.Series,
    benchmarks: GlobalBenchmarks,
    cfg: PrioritizationConfig,
) -> tuple[str, str, dict[str, int]]:
    """Score each lever for one segment and return primary + secondary + all scores.

    Four levers with additive scoring rules (preserved exactly from notebook):

        parts_logistics:
            +2 if parts_rate > global × 1.15
            +3 if delivery_days_median > global × 1.30 (if not NaN)

        engineer_deployment:
            +3 if engineer_q4_rate > global × 1.30 (strong)
            +1 elif engineer_q4_rate > global × 1.10 (weak)

        channel_process:
            score = max(0, ch_risk_ordinal - 3)
            (DMS=0, DMS2=0, ASC=0, Premier=1, ASD=2, AE=3, SPO=4)

        repair_complexity:
            +2 if sealed_rate > global × 1.30
            +1 if reclaim_rate > global × 1.30

    Tie-breaking: ``max()`` returns the first lever found at the max
    score; ``sorted(... reverse=True)`` picks the second from the same
    sort. This matches the notebook's behavior exactly.

    Returns:
        (primary_lever, secondary_lever, all_scores_dict)
    """
    scores: dict[str, int] = {}

    # Parts lever
    parts_score = 0
    if row["parts_rate"] > benchmarks.parts_rate * cfg.parts_rate_mult:
        parts_score += 2
    if not np.isnan(row["delivery_days_median"]):
        if row["delivery_days_median"] > benchmarks.parts_delivery_days * cfg.parts_delivery_mult:
            parts_score += 3
    scores["parts_logistics"] = parts_score

    # Engineer lever
    eng_score = 0
    if row["engineer_q4_rate"] > benchmarks.engineer_q4_rate * cfg.engineer_q4_strong_mult:
        eng_score += 3
    elif row["engineer_q4_rate"] > benchmarks.engineer_q4_rate * cfg.engineer_q4_weak_mult:
        eng_score += 1
    scores["engineer_deployment"] = eng_score

    # Channel / process lever
    ch_risk = CHANNEL_MAP.get(row["Channel"], 3)
    scores["channel_process"] = max(0, ch_risk - 3)

    # Repair complexity lever
    cx_score = 0
    if row["sealed_rate"] > benchmarks.sealed_rate * cfg.sealed_rate_mult:
        cx_score += 2
    if row["reclaim_rate"] > benchmarks.reclaim_rate * cfg.reclaim_rate_mult:
        cx_score += 1
    scores["repair_complexity"] = cx_score

    primary = max(scores, key=scores.get)
    secondary = sorted(scores, key=scores.get, reverse=True)[1]
    return primary, secondary, scores


def build_lever_decomposition(
    priority_df: pd.DataFrame,
    benchmarks: GlobalBenchmarks,
    cfg: PrioritizationConfig,
) -> pd.DataFrame:
    """Apply lever scoring to every segment in the priority matrix.

    Returns a long-form table with primary_lever, secondary_lever, and
    all four individual lever scores for each segment, plus the underlying
    lever-indicator values (parts_rate, engineer_q4_rate, delivery_days_median).
    """
    lever_rows: list[dict] = []
    for _, row in priority_df.iterrows():
        primary, secondary, scores = classify_primary_lever(row, benchmarks, cfg)
        lever_rows.append({
            "Market_Category": row["Market_Category"],
            "Channel": row["Channel"],
            "n_repairs": row["n_repairs"],
            "actual_late_5": row["actual_late_5"],
            "cases_at_risk_5": row["cases_at_risk_5"],
            "primary_lever": primary,
            "secondary_lever": secondary,
            "parts_score": scores["parts_logistics"],
            "engineer_score": scores["engineer_deployment"],
            "channel_score": scores["channel_process"],
            "complexity_score": scores["repair_complexity"],
            "parts_rate": row["parts_rate"],
            "engineer_q4_rate": row["engineer_q4_rate"],
            "delivery_days_median": row["delivery_days_median"],
        })
    return pd.DataFrame(lever_rows)


# =====================================================================
# SECTION 8E: THRESHOLD SHIFT ANALYSIS
# =====================================================================

def build_threshold_shift(
    priority_df: pd.DataFrame,
    cfg: PrioritizationConfig,
) -> tuple[dict[int, pd.DataFrame], set, set]:
    """Show how segment priority changes across T = 3, 5, 7, 10.

    For each threshold, ranks segments by ``n_repairs × actual_late_{T}``
    (cases at risk) and returns the top 10. Also computes:
        - ``consistent``: segments in top 10 across all four thresholds
        - ``strict_only``: in top 10 at T=3 but not at T=10

    Returns:
        (threshold_rankings, consistent_segments, strict_only_segments)
    """
    threshold_rankings: dict[int, pd.DataFrame] = {}
    top10_per_threshold: dict[int, set] = {}

    for t in cfg.ontime_targets:
        ranked = priority_df.copy()
        ranked[f"impact_{t}"] = (
            ranked["n_repairs"] * ranked[f"actual_late_{t}"]
        ).astype(int)
        ranked[f"rank_{t}"] = ranked[f"impact_{t}"].rank(
            ascending=False, method="min",
        ).astype(int)
        threshold_rankings[t] = ranked.nsmallest(10, f"rank_{t}")[
            ["Market_Category", "Channel",
             f"actual_late_{t}", f"cases_at_risk_{t}", f"rank_{t}"]
        ]
        top10_per_threshold[t] = set(
            zip(ranked.nsmallest(10, f"rank_{t}")["Market_Category"],
                ranked.nsmallest(10, f"rank_{t}")["Channel"])
        )

    # Consistent across all thresholds
    consistent = top10_per_threshold[cfg.ontime_targets[0]]
    for t in cfg.ontime_targets[1:]:
        consistent = consistent & top10_per_threshold[t]

    # Strict-only (T=3 only, falls out by T=10)
    strict_only = top10_per_threshold[cfg.ontime_targets[0]] - top10_per_threshold[cfg.ontime_targets[-1]]

    return threshold_rankings, consistent, strict_only


def build_threshold_shift_table(
    threshold_rankings: dict[int, pd.DataFrame],
    cfg: PrioritizationConfig,
) -> pd.DataFrame:
    """Flatten the per-threshold rankings into one long-form CSV-ready table."""
    rows: list[dict] = []
    for t, df in threshold_rankings.items():
        for _, r in df.iterrows():
            rows.append({
                "threshold": t,
                "Market_Category": r["Market_Category"],
                "Channel": r["Channel"],
                "actual_late": r[f"actual_late_{t}"],
                "cases_at_risk": r[f"cases_at_risk_{t}"],
                "rank": r[f"rank_{t}"],
            })
    return pd.DataFrame(rows)


# =====================================================================
# SECTION 8F: TIER-LEVEL SUMMARY
# =====================================================================

def build_tier_summary(
    full_df: pd.DataFrame,
    lever_df: pd.DataFrame,
    cfg: PrioritizationConfig,
) -> pd.DataFrame:
    """Aggregate up to Market_Category level: volume, late rates, dominant lever.

    Used to estimate "tier elevation" improvement potential in the final
    recommendation table.
    """
    rows: list[dict] = []
    n_total = len(full_df)
    for tier in cfg.tier_order:
        g = full_df[full_df["Market_Category"] == tier]
        if len(g) == 0:
            continue
        row = {
            "Market_Category": tier,
            "n_repairs": len(g),
            "pct_of_total": round(len(g) / n_total, 4),
            "mean_rtat": round(float(g["target_days"].mean()), 2),
            "mean_pred_rtat": round(float(g["pred_rtat"].mean()), 2),
        }
        for t in cfg.ontime_targets:
            late = 1 - g[f"OnTime_{t}"].astype(float).mean()
            row[f"late_rate_{t}"] = round(float(late), 4)
            row[f"cases_at_risk_{t}"] = int(len(g) * late)

        tier_levers = lever_df[lever_df["Market_Category"] == tier]
        row["primary_lever"] = (
            tier_levers["primary_lever"].mode().iloc[0]
            if len(tier_levers) > 0 else "unknown"
        )
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# SECTION 8G: NPS POST-HOC VALIDATION
# =====================================================================

def build_nps_validation(
    full_df: pd.DataFrame,
    cfg: PrioritizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that high predicted-risk segments have lower NPS.

    NPS signals (``is_promoter``, ``is_detractor``) were intentionally
    excluded from the model. If prioritization is operationally correct,
    segments with high ``pred_late_prob_5`` should have lower promoter
    rates and higher detractor rates among NPS responders.

    Two tables produced:
        - NPS by predicted-risk bucket (5 bins)
        - NPS by Market_Category

    Uses only the configured ``nps_validation_year`` (2025) for best
    coverage. Returns (nps_by_risk, nps_by_tier).
    """
    nps_sub = full_df[
        (full_df["source_year"] == cfg.nps_validation_year)
        & (full_df["has_nps"] == 1)
    ].copy()

    if len(nps_sub) == 0:
        logger.warning("No NPS responders found for year %d", cfg.nps_validation_year)
        return pd.DataFrame(), pd.DataFrame()

    logger.info("NPS validation cohort (%d responders): %s",
                cfg.nps_validation_year, f"{len(nps_sub):,}")

    nps_sub["pred_risk_bucket"] = pd.cut(
        nps_sub["pred_late_prob_5"],
        bins=list(cfg.nps_risk_bins),
        labels=list(cfg.nps_risk_labels),
    )

    nps_by_risk = (
        nps_sub.groupby("pred_risk_bucket", observed=True)
        .agg(
            n=("is_promoter", "count"),
            promoter_rate=("is_promoter", "mean"),
            detractor_rate=("is_detractor", "mean"),
            mean_pred_rtat=("pred_rtat", "mean"),
        )
        .round(3)
    )

    nps_by_tier = (
        nps_sub.groupby("Market_Category")
        .agg(
            n=("is_promoter", "count"),
            promoter_rate=("is_promoter", "mean"),
            detractor_rate=("is_detractor", "mean"),
            mean_rtat=("target_days", "mean"),
            mean_pred_late=("pred_late_prob_5", "mean"),
        )
        .round(3)
        .sort_values("promoter_rate", ascending=False)
    )

    return nps_by_risk, nps_by_tier


# =====================================================================
# SECTION 8H: FINAL RECOMMENDATION TABLE
# =====================================================================

def _lever_recommendation_text(row: pd.Series) -> str:
    """Generate stakeholder-readable action text for a segment's primary lever."""
    lever = row["primary_lever"]
    if lever == "engineer_deployment":
        return (
            f"Redeploy Q4 engineers ({row['engineer_q4_rate']:.0%} of "
            f"repairs assigned to slowest quartile). Target Q1/Q2 "
            f"engineer assignment."
        )
    if lever == "parts_logistics":
        days = (f"{row['delivery_days_median']:.1f}d"
                if not np.isnan(row["delivery_days_median"]) else "unknown")
        return (
            f"Reduce parts delivery time (median {days}). Prioritize "
            f"overnight shipping or pre-position high-demand parts regionally."
        )
    if lever == "channel_process":
        return (
            f"Channel process improvement in {row['Channel']}. Review "
            f"scheduling, dispatch, and SLA compliance."
        )
    # repair_complexity
    return (
        "Reduce repair complexity burden. Improve first-time fix rate "
        "and reclaim prevention."
    )


def build_final_recommendation(
    lever_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    tier_sum_df: pd.DataFrame,
    cfg: PrioritizationConfig,
) -> pd.DataFrame:
    """Merge lever_df with priority_df, attach action text + tier-elevation potential.

    The "tier elevation" estimate (``cases_improvable``) projects how
    many at-risk cases would be saved if the segment's late rate
    matched the next-better tier's average. This is a back-of-envelope
    upper bound on operational improvement potential, not a forecast.
    """
    cols_to_add = [
        c for c in (
            "actual_late_3", "actual_late_5",
            "actual_late_7", "actual_late_10",
            "pred_late_5", "mean_rtat_actual",
            "priority_rank_5",
        )
        if c not in lever_df.columns
    ]
    final_rec = lever_df.merge(
        priority_df[["Market_Category", "Channel"] + cols_to_add],
        on=["Market_Category", "Channel"], how="left",
    )

    final_rec["recommendation"] = final_rec.apply(
        _lever_recommendation_text, axis=1,
    )

    # Tier elevation improvement estimate
    tier_late5 = (
        tier_sum_df.set_index("Market_Category")["late_rate_5"].to_dict()
        if len(tier_sum_df) > 0 else {}
    )

    def expected_improvement(row: pd.Series) -> int:
        """Estimate cases improvable if segment matched next-better tier."""
        tier = row["Market_Category"]
        idx = cfg.tier_order.index(tier) if tier in cfg.tier_order else -1
        if idx <= 0:
            return 0
        better_tier = cfg.tier_order[idx - 1]
        target_late = tier_late5.get(better_tier, row["actual_late_5"] * 0.8)
        improvement = max(0, row["actual_late_5"] - target_late)
        return int(improvement * row["n_repairs"])

    final_rec["cases_improvable"] = final_rec.apply(expected_improvement, axis=1)
    final_rec = final_rec.sort_values("cases_at_risk_5", ascending=False).reset_index(drop=True)
    final_rec["overall_rank"] = final_rec.index + 1
    return final_rec


# =====================================================================
# MAIN ORCHESTRATION
# =====================================================================

def run_prioritization(cfg: PrioritizationConfig | None = None) -> dict:
    """Run the full Step 8 prioritization pipeline end-to-end.

    Writes all CSV deliverables under ``cfg.output_dir``. Returns a
    dict of intermediate result tables for inspection.
    """
    cfg = cfg or PrioritizationConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 8A: load ----
    logger.info("=== 8A: Loading models and data ===")
    arts = load_models_and_data(cfg)
    full_df = merge_context(arts["train_feat"], arts["master_train"])
    hold_df = merge_context(arts["holdout_feat"], arts["master_hold"])
    logger.info("  Full train+val: %s | Holdout: %s",
                f"{len(full_df):,}", f"{len(hold_df):,}")

    # ---- 8B: attach predictions ----
    logger.info("=== 8B: Generating predictions ===")
    full_df = attach_predictions(
        full_df, arts["lgb_models"], arts["lgb_reg"], cfg.ontime_targets,
    )
    hold_df = attach_predictions(
        hold_df, arts["lgb_models"], arts["lgb_reg"], cfg.ontime_targets,
    )

    # ---- 8C: priority matrix ----
    logger.info("=== 8C: Building priority matrix ===")
    priority_df = build_priority_matrix(full_df, cfg)
    priority_df.to_csv(cfg.output_dir / "priority_matrix.csv", index=False)
    logger.info("  Segments (min %d repairs): %d",
                cfg.min_segment_repairs, len(priority_df))

    # ---- 8D: lever decomposition (4 levers, primary + secondary) ----
    logger.info("=== 8D: Lever decomposition ===")
    benchmarks = compute_global_benchmarks(full_df)
    lever_df = build_lever_decomposition(priority_df, benchmarks, cfg)
    lever_df.to_csv(cfg.output_dir / "lever_decomposition.csv", index=False)
    for lever, count in lever_df["primary_lever"].value_counts().items():
        pct = count / len(lever_df)
        logger.info("  %-25s: %d segments (%.0f%%)", lever, count, 100 * pct)

    # ---- 8E: threshold shift ----
    logger.info("=== 8E: Threshold shift analysis ===")
    threshold_rankings, consistent, strict_only = build_threshold_shift(
        priority_df, cfg,
    )
    threshold_shift_table = build_threshold_shift_table(threshold_rankings, cfg)
    threshold_shift_table.to_csv(cfg.output_dir / "threshold_shift.csv", index=False)
    logger.info("  Consistent (top 10 across T=3,5,7,10): %d segments",
                len(consistent))

    # ---- 8F: tier summary ----
    logger.info("=== 8F: Tier-level summary ===")
    tier_sum_df = build_tier_summary(full_df, lever_df, cfg)
    tier_sum_df.to_csv(cfg.output_dir / "tier_summary.csv", index=False)

    # ---- 8G: NPS validation ----
    logger.info("=== 8G: NPS post-hoc validation ===")
    nps_by_risk, nps_by_tier = build_nps_validation(full_df, cfg)
    if not nps_by_risk.empty:
        nps_by_risk.to_csv(cfg.output_dir / "nps_validation.csv")
        nps_by_tier.to_csv(cfg.output_dir / "nps_by_tier.csv")
        # Print headline gap
        try:
            promoter_low = float(nps_by_risk.iloc[0]["promoter_rate"])
            promoter_high = float(nps_by_risk.iloc[-1]["promoter_rate"])
            detractor_low = float(nps_by_risk.iloc[0]["detractor_rate"])
            detractor_high = float(nps_by_risk.iloc[-1]["detractor_rate"])
            logger.info(
                "  Promoter gap: very-low=%.1f%% → very-high=%.1f%% (Δ %.1fpp)",
                100 * promoter_low, 100 * promoter_high,
                100 * (promoter_low - promoter_high),
            )
            logger.info(
                "  Detractor gap: very-low=%.1f%% → very-high=%.1f%% (Δ %.1fpp)",
                100 * detractor_low, 100 * detractor_high,
                100 * (detractor_high - detractor_low),
            )
        except (IndexError, KeyError):
            pass

    # ---- 8H: final recommendation ----
    logger.info("=== 8H: Final recommendation ===")
    final_rec = build_final_recommendation(lever_df, priority_df, tier_sum_df, cfg)
    final_rec.to_csv(cfg.output_dir / "final_recommendation.csv", index=False)
    logger.info("  Top %d segments saved", cfg.top_n_segments)

    return {
        "priority_matrix": priority_df,
        "lever_decomposition": lever_df,
        "threshold_rankings": threshold_rankings,
        "threshold_shift": threshold_shift_table,
        "consistent_segments": consistent,
        "strict_only_segments": strict_only,
        "tier_summary": tier_sum_df,
        "nps_by_risk": nps_by_risk,
        "nps_by_tier": nps_by_tier,
        "final_recommendation": final_rec,
        "benchmarks": benchmarks,
    }


def _configure_logging() -> None:
    """Configure root logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    results = run_prioritization()
    logger.info(
        "Prioritization complete. %d segments analyzed; top recommendation: %s × %s",
        len(results["priority_matrix"]),
        results["final_recommendation"].iloc[0]["Market_Category"],
        results["final_recommendation"].iloc[0]["Channel"],
    )
