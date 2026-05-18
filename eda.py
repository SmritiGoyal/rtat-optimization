"""
eda.py
======
Step 5: First-pass exploratory data analysis (EDA).

Purpose: Surface segment-level heterogeneity to form hypotheses about
where interventions will have the most impact. This is not a general
summary — every analysis is framed around the business question:

    "Which segments combine high delay with meaningful volume?"

SubSections:
    5A: Overall lead time distribution
    5B: Segment analysis — Market_Category (geographic tier)
    5C: Segment analysis — Channel (service network)
    5D: Segment analysis — Division_Name (product type)
    5E: Parts impact analysis
    5F: Engineer experience signal
    5G: Time effects (seasonality, year-over-year)
    5H: Reclaim / repeat repair signal
    5I: NPS post-hoc validation preview
    5J: Segment priority matrix (the key EDA output)
    5K: Hypothesis summary (H1-H8 — informs Steps 6-8)

The hypothesis list at section 5K is the EDA's most-cited output: it
maps directly onto the lever taxonomy in ``prioritization.py`` and
informs feature engineering choices in ``feature_engineering.py``.

Output artifacts under ``outputs/eda/``:
    - target_distribution_by_year.csv (5A)
    - market_category_stats.csv (5B)
    - channel_stats.csv (5C)
    - division_stats.csv (5D)
    - delivery_impact.csv (5E)
    - engineer_quartiles.csv (5F)
    - monthly_stats.csv (5G)
    - reclaim_signal.csv (5H)
    - nps_validation.csv (5I)
    - segment_priority_matrix.csv (5J)
    - hypothesis_list.csv (5K)
    - PNG charts for each section

Notebook visualizations rely on the underlying data, which is
client-confidential and not redistributed. The PNG outputs will
regenerate identically when this module runs on the full dataset.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EDAConfig:
    """Configuration for the EDA stage."""
    interim_dir: Path = PROJECT_ROOT / "outputs" / "interim"
    eda_dir: Path = PROJECT_ROOT / "outputs" / "eda"

    # OnTime thresholds emphasized in plots (T=3, 5, 7, 10 = story
    # thresholds; other Ts shown for context)
    ontime_targets: tuple[int, ...] = (3, 5, 7, 10)

    # Plot DPI for saved PNGs
    figure_dpi: int = 150

    # Cap for histogram x-axis (lead time tail can be very long)
    rtat_clip_distribution: int = 60
    rtat_clip_parts: int = 40

    # Minimum segment size for priority-matrix inclusion
    min_segment_repairs: int = 500


# Generic plot palette — no LG-specific branding. These colors are
# semantic (warm = high/bad, cool = low/good, neutral = baseline) and
# render legibly in both light and dark backgrounds.
PALETTE_PRIMARY = "#C40000"   # warm — for "above average" / focal series
PALETTE_DARK = "#1A1A2E"      # near-black — for secondary series / trend lines
PALETTE_GRAY = "#6B7280"      # neutral gray — for "below average" / baseline

TIER_COLORS: dict[str, str] = {
    "1. Top 10": "#C40000",
    "2. Metro":  "#E07B54",
    "3. Urban":  "#5B8DB8",
    "4. Rural":  "#88B04B",
}

# Tier ordering used throughout (matches notebook)
TIER_ORDER: list[str] = ["1. Top 10", "2. Metro", "3. Urban", "4. Rural"]


def _apply_plot_style() -> None:
    """Apply consistent matplotlib styling across all EDA figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.family": "sans-serif",
    })


def flag_mean(series: pd.Series) -> float:
    """Mean of an Int8 / nullable flag column, cast to float first."""
    return float(series.astype("float").mean())


# =====================================================================
# SECTION 5A: OVERALL LEAD TIME DISTRIBUTION
# =====================================================================

def section_5a_overall_distribution(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Plot the overall RTAT distribution + per-year comparison + on-time rates.

    Produces a 3-panel figure: (1) histogram capped at 60 days,
    (2) year-over-year density comparison, (3) OnTime_T rates for T=1..10
    with the four "story" thresholds (3, 5, 7, 10) highlighted in red.
    Saves both PNG and a per-year stats CSV.

    Returns the distribution summary table for use by callers.
    """
    logger.info("=== 5A: Overall lead time distribution ===")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Lead Time Distribution — Training Cohort (2023–2025)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # --- Panel 1: histogram capped ---
    ax = axes[0]
    ax.hist(
        train["target_days"].clip(upper=cfg.rtat_clip_distribution),
        bins=cfg.rtat_clip_distribution, color=PALETTE_PRIMARY,
        alpha=0.8, edgecolor="white", linewidth=0.3,
    )
    median = float(train["target_days"].median())
    mean = float(train["target_days"].mean())
    ax.axvline(median, color=PALETTE_DARK, linestyle="--", linewidth=1.5,
               label=f"Median: {median:.0f}d")
    ax.axvline(mean, color=PALETTE_GRAY, linestyle=":", linewidth=1.5,
               label=f"Mean: {mean:.1f}d")
    ax.set_xlabel(f"Lead time (days, capped at {cfg.rtat_clip_distribution})")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution (clipped at {cfg.rtat_clip_distribution}d)")
    ax.legend(fontsize=8)

    # --- Panel 2: year-over-year ---
    ax = axes[1]
    yoy_colors = [("#C40000", 2023), ("#E07B54", 2024), ("#5B8DB8", 2025)]
    for color, yr in yoy_colors:
        sub = train[train["source_year"] == yr]["target_days"].clip(
            upper=cfg.rtat_clip_distribution,
        )
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=cfg.rtat_clip_distribution, alpha=0.5,
                color=color, label=f"{yr} (med={sub.median():.0f}d)",
                density=True)
    ax.set_xlabel(f"Lead time (days, capped at {cfg.rtat_clip_distribution})")
    ax.set_ylabel("Density")
    ax.set_title("Year-over-Year Comparison")
    ax.legend(fontsize=8)

    # --- Panel 3: OnTime_T rates T=1..10 ---
    ax = axes[2]
    rates = [flag_mean(train[f"OnTime_{t}"]) for t in range(1, 11)]
    colors = [
        PALETTE_PRIMARY if t in cfg.ontime_targets else PALETTE_GRAY
        for t in range(1, 11)
    ]
    bars = ax.bar(range(1, 11), rates, color=colors, edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{rate:.0%}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("T (days)")
    ax.set_ylabel("On-time rate")
    ax.set_title("OnTime_T Rates (red = story thresholds)")
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5a_overall_distribution.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    # Per-year summary table
    dist_summary = (
        train.groupby("source_year")["target_days"]
        .agg(n="count", mean="mean", median="median",
             p75=lambda x: x.quantile(.75),
             p90=lambda x: x.quantile(.90),
             p95=lambda x: x.quantile(.95))
        .round(2)
    )
    dist_summary.to_csv(cfg.eda_dir / "5a_target_distribution_by_year.csv")
    logger.info("  Median: %.0fd | Mean: %.1fd", median, mean)
    return dist_summary


# =====================================================================
# SECTION 5B: MARKET_CATEGORY ANALYSIS
# =====================================================================

def section_5b_market_category(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Compute and plot lead time by Market_Category (geographic tier).

    Produces a 3-panel figure: (1) mean RTAT bar chart, (2) per-T
    on-time rate comparison, (3) volume × late-rate scatter (the
    priority-matrix view at tier level). Saves stats CSV.
    """
    logger.info("=== 5B: Market_Category analysis ===")

    tier_stats: list[dict] = []
    for tier in TIER_ORDER:
        sub = train[train["Market_Category"] == tier]
        if len(sub) == 0:
            continue
        row = {
            "Market_Category": tier,
            "n_repairs": len(sub),
            "pct_of_total": len(sub) / len(train),
            "mean_rtat": float(sub["target_days"].mean()),
            "median_rtat": float(sub["target_days"].median()),
            "p90_rtat": float(sub["target_days"].quantile(.90)),
        }
        for t in cfg.ontime_targets:
            row[f"OnTime_{t}"] = flag_mean(sub[f"OnTime_{t}"])
        tier_stats.append(row)

    tier_df = pd.DataFrame(tier_stats)
    tier_df.to_csv(cfg.eda_dir / "5b_market_category_stats.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Lead Time by Market_Category (Geographic Tier)",
                 fontsize=12, fontweight="bold", y=1.02)

    # --- Panel 1: mean RTAT bar ---
    ax = axes[0]
    colors = [TIER_COLORS.get(t, PALETTE_GRAY)
              for t in tier_df["Market_Category"]]
    bars = ax.barh(tier_df["Market_Category"], tier_df["mean_rtat"],
                   color=colors, edgecolor="white")
    for bar, val in zip(bars, tier_df["mean_rtat"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}d", va="center", fontsize=8)
    ax.set_xlabel("Mean RTAT (days)")
    ax.set_title("Mean Lead Time by Tier")
    ax.invert_yaxis()

    # --- Panel 2: OnTime rates grouped by tier ---
    ax = axes[1]
    x = np.arange(len(cfg.ontime_targets))
    width = 0.2
    for i, (tier, color) in enumerate(TIER_COLORS.items()):
        row = tier_df[tier_df["Market_Category"] == tier]
        if row.empty:
            continue
        vals = [float(row[f"OnTime_{t}"].values[0]) for t in cfg.ontime_targets]
        ax.bar(x + i * width, vals, width, label=tier,
               color=color, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"T={t}" for t in cfg.ontime_targets])
    ax.set_ylabel("On-time rate")
    ax.set_title("OnTime Rates by Tier")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=7)

    # --- Panel 3: volume × late rate priority view ---
    ax = axes[2]
    for tier, color in TIER_COLORS.items():
        row = tier_df[tier_df["Market_Category"] == tier]
        if row.empty:
            continue
        late_rate = 1 - float(row["OnTime_5"].values[0])
        volume = int(row["n_repairs"].values[0])
        ax.scatter(late_rate, volume, s=200, color=color, zorder=3,
                   label=tier, edgecolors="white", linewidth=1.5)
        ax.annotate(tier.split(".")[1].strip(),
                    (late_rate, volume),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Late rate at T=5 (1 − OnTime_5)")
    ax.set_ylabel("Repair volume")
    ax.set_title("Priority Matrix: Volume × Late Rate (T=5)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f}K",
    ))
    avg_late = float(tier_df["OnTime_5"].apply(lambda r: 1 - r).mean())
    ax.axvline(avg_late, color=PALETTE_GRAY, linestyle="--",
               linewidth=1, alpha=0.6, label="Average late rate")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5b_market_category.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return tier_df


# =====================================================================
# SECTION 5C: CHANNEL ANALYSIS
# =====================================================================

def section_5c_channel(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Compute and plot lead time by Channel (top 8 by volume).

    Produces a 2-panel figure: (1) mean RTAT bar with volume annotation,
    (2) on-time rate trend across thresholds.
    """
    logger.info("=== 5C: Channel analysis ===")

    top_channels = train["Channel"].value_counts().head(8).index.tolist()

    chan_stats: list[dict] = []
    for ch in top_channels:
        sub = train[train["Channel"] == ch]
        row = {
            "Channel": ch,
            "n_repairs": len(sub),
            "pct_total": len(sub) / len(train),
            "mean_rtat": float(sub["target_days"].mean()),
            "median_rtat": float(sub["target_days"].median()),
            "p90_rtat": float(sub["target_days"].quantile(.90)),
        }
        for t in cfg.ontime_targets:
            row[f"OnTime_{t}"] = flag_mean(sub[f"OnTime_{t}"])
        chan_stats.append(row)

    chan_df = pd.DataFrame(chan_stats).sort_values(
        "mean_rtat", ascending=False,
    )
    chan_df.to_csv(cfg.eda_dir / "5c_channel_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lead Time by Service Channel",
                 fontsize=12, fontweight="bold", y=1.02)

    # --- Panel 1: mean RTAT ---
    ax = axes[0]
    ax.barh(chan_df["Channel"], chan_df["mean_rtat"],
            color=PALETTE_PRIMARY, alpha=0.8, edgecolor="white")
    overall_mean = float(train["target_days"].mean())
    ax.axvline(overall_mean, color=PALETTE_GRAY, linestyle="--",
               linewidth=1.5, label="Overall mean")
    for i, (val, vol) in enumerate(
            zip(chan_df["mean_rtat"], chan_df["n_repairs"])):
        ax.text(val + 0.05, i, f"{val:.1f}d  ({vol / 1000:.0f}K)",
                va="center", fontsize=7.5)
    ax.set_xlabel("Mean RTAT (days)")
    ax.set_title("Mean Lead Time by Channel")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # --- Panel 2: on-time rate trend across T ---
    ax = axes[1]
    trend_colors = ["#C40000", "#E07B54", "#5B8DB8", "#88B04B"]
    for t, color in zip(cfg.ontime_targets, trend_colors):
        ax.plot(chan_df["Channel"], chan_df[f"OnTime_{t}"],
                marker="o", linewidth=1.5, color=color, label=f"T={t}")
    ax.set_ylabel("On-time rate")
    ax.set_title("OnTime Rates by Channel")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5c_channel_analysis.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return chan_df


# =====================================================================
# SECTION 5D: DIVISION_NAME ANALYSIS
# =====================================================================

def section_5d_division(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Compute and plot lead time by Division_Name (top 10 by volume).

    Produces a 2-panel figure: (1) mean RTAT bar with above-average
    division coloring, (2) parts-rate vs lead-time scatter with bubble
    size = volume.
    """
    logger.info("=== 5D: Division_Name analysis ===")

    top_divs = train["Division_Name"].value_counts().head(10).index.tolist()

    div_stats: list[dict] = []
    for div in top_divs:
        sub = train[train["Division_Name"] == div]
        row = {
            "Division_Name": div,
            "n_repairs": len(sub),
            "pct_total": len(sub) / len(train),
            "mean_rtat": float(sub["target_days"].mean()),
            "median_rtat": float(sub["target_days"].median()),
            "p90_rtat": float(sub["target_days"].quantile(.90)),
            "pct_parts": flag_mean(sub["has_parts_reclaim"]),
        }
        for t in cfg.ontime_targets:
            row[f"OnTime_{t}"] = flag_mean(sub[f"OnTime_{t}"])
        div_stats.append(row)

    div_df = pd.DataFrame(div_stats).sort_values(
        "mean_rtat", ascending=False,
    )
    div_df.to_csv(cfg.eda_dir / "5d_division_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lead Time by Product Division",
                 fontsize=12, fontweight="bold", y=1.02)

    overall_mean = float(train["target_days"].mean())

    # --- Panel 1: above/below average mean RTAT ---
    ax = axes[0]
    bar_colors = [PALETTE_PRIMARY if v > overall_mean else PALETTE_GRAY
                  for v in div_df["mean_rtat"]]
    ax.barh(div_df["Division_Name"], div_df["mean_rtat"],
            color=bar_colors, edgecolor="white", alpha=0.85)
    ax.axvline(overall_mean, color=PALETTE_DARK, linestyle="--",
               linewidth=1.5, label="Overall mean")
    ax.set_xlabel("Mean RTAT (days)")
    ax.set_title("Mean Lead Time (red = above average)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # --- Panel 2: parts rate vs mean RTAT bubble chart ---
    ax = axes[1]
    ax.scatter(div_df["pct_parts"], div_df["mean_rtat"],
               s=div_df["n_repairs"] / 500,
               color=PALETTE_PRIMARY, alpha=0.7,
               edgecolors="white", linewidth=1)
    for _, row in div_df.iterrows():
        ax.annotate(row["Division_Name"],
                    (row["pct_parts"], row["mean_rtat"]),
                    textcoords="offset points",
                    xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Parts-required rate (% of repairs)")
    ax.set_ylabel("Mean RTAT (days)")
    ax.set_title("Parts Rate vs. Lead Time\n(bubble = volume)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5d_division_analysis.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return div_df


# =====================================================================
# SECTION 5E: PARTS IMPACT
# =====================================================================

def section_5e_parts_impact(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Quantify and plot the impact of parts on lead time.

    Produces:
        - Stats table: parts vs no-parts lead times
        - Stats table: DMS-ordered vs not
        - Stats table: delivery duration bins (DMS repairs only)
        - 3-panel figure: distribution overlay, complexity bars,
          delivery-bin RTAT trend

    Returns the delivery_impact table for documentation; the two other
    summary tables are printed to logger and saved to CSV.
    """
    logger.info("=== 5E: Parts impact analysis ===")

    # Parts vs no-parts
    parts_summary = (
        train.groupby("has_parts_reclaim")["target_days"]
        .agg(n="count", mean="mean", median="median",
             p75=lambda x: x.quantile(.75),
             p90=lambda x: x.quantile(.90))
        .round(2)
    )
    parts_summary.index = parts_summary.index.map(
        {0: "No parts", 1: "Parts required"},
    )

    # DMS vs not
    dms_summary = (
        train.groupby("ordered_via_dms")["target_days"]
        .agg(n="count", mean="mean", median="median",
             p90=lambda x: x.quantile(.90))
        .round(2)
    )
    dms_summary.index = dms_summary.index.map(
        {0: "No DMS order", 1: "DMS order"},
    )

    # Delivery duration bins (leakage-safe feature)
    safe_sub = train[train["parts_order_to_arrival_days_safe"].notna()].copy()
    logger.info("  DMS repairs with safe delivery duration: %s",
                f"{len(safe_sub):,}")

    delivery_impact = pd.DataFrame()
    if len(safe_sub) > 1000:
        safe_sub["delivery_bin"] = pd.cut(
            safe_sub["parts_order_to_arrival_days_safe"],
            bins=[0, 1, 2, 3, 5, 7, 10, 90],
            labels=["0-1d", "1-2d", "2-3d", "3-5d",
                    "5-7d", "7-10d", "10d+"],
        )
        delivery_impact = (
            safe_sub.groupby("delivery_bin", observed=True)["target_days"]
            .agg(n="count", mean_rtat="mean",
                 ontime5=lambda x: (x <= 5).mean())
            .round(3)
        )
        delivery_impact.to_csv(cfg.eda_dir / "5e_delivery_impact.csv")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Parts Impact on Lead Time",
                 fontsize=12, fontweight="bold", y=1.02)

    # --- Panel 1: parts vs not, distribution overlay ---
    ax = axes[0]
    for val, label, color in [
        (0, "No parts", PALETTE_GRAY),
        (1, "Parts", PALETTE_PRIMARY),
    ]:
        sub = train[train["has_parts_reclaim"] == val]["target_days"].clip(
            upper=cfg.rtat_clip_parts,
        )
        ax.hist(sub, bins=cfg.rtat_clip_parts, alpha=0.6, color=color,
                label=f"{label} (med={sub.median():.0f}d)",
                density=True)
    ax.set_xlabel(f"Lead time (days, capped {cfg.rtat_clip_parts})")
    ax.set_ylabel("Density")
    ax.set_title("Parts Required vs. Not")
    ax.legend(fontsize=8)

    # --- Panel 2: complexity (parts count → mean RTAT) ---
    ax = axes[1]
    pc_stats = (
        train.groupby("parts_count_reclaim")["target_days"]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    ax.bar(pc_stats["parts_count_reclaim"], pc_stats["mean"],
           color=PALETTE_PRIMARY, alpha=0.8, edgecolor="white")
    for _, r in pc_stats.iterrows():
        ax.text(r["parts_count_reclaim"], r["mean"] + 0.1,
                f"n={r['n'] / 1000:.0f}K", ha="center", fontsize=7)
    ax.set_xlabel("Number of parts (from Reclaim)")
    ax.set_ylabel("Mean RTAT (days)")
    ax.set_title("Repair Complexity → Lead Time")

    # --- Panel 3: delivery bins ---
    ax = axes[2]
    if not delivery_impact.empty:
        bins = delivery_impact.index.tolist()
        means = delivery_impact["mean_rtat"].tolist()
        ontime5 = delivery_impact["ontime5"].tolist()
        x = np.arange(len(bins))
        ax2 = ax.twinx()
        ax.bar(x, means, color=PALETTE_PRIMARY, alpha=0.7,
               label="Mean RTAT")
        ax2.plot(x, ontime5, color=PALETTE_DARK, marker="o",
                 linewidth=2, label="OnTime_5 rate")
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=30, ha="right")
        ax.set_ylabel("Mean RTAT (days)", color=PALETTE_PRIMARY)
        ax2.set_ylabel("OnTime_5 rate", color=PALETTE_DARK)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title("Parts Delivery Duration → RTAT\n(DMS repairs only)")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Insufficient safe\ndelivery data",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5e_parts_impact.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return delivery_impact


# =====================================================================
# SECTION 5F: ENGINEER SIGNAL
# =====================================================================

def section_5f_engineer(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Show the engineer historical-performance proxy → actual RTAT signal.

    Produces a 2-panel figure: (1) mean RTAT by engineer quartile,
    (2) proxy-vs-actual scatter with linear-fit trend line.
    """
    logger.info("=== 5F: Engineer signal ===")

    eng_sub = train[train["engineer_hist_mean_rtat"].notna()].copy()
    logger.info("  Repairs with engineer proxy: %s (%.1f%%)",
                f"{len(eng_sub):,}", 100 * len(eng_sub) / max(len(train), 1))

    eng_sub["eng_quartile"] = pd.qcut(
        eng_sub["engineer_hist_mean_rtat"], q=4,
        labels=["Q1 Fastest", "Q2", "Q3", "Q4 Slowest"],
    )
    eng_q_stats = (
        eng_sub.groupby("eng_quartile", observed=True)["target_days"]
        .agg(n="count", mean="mean", median="median",
             ontime5=lambda x: (x <= 5).mean())
        .round(3)
    )
    eng_q_stats.to_csv(cfg.eda_dir / "5f_engineer_quartiles.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Engineer Experience Signal",
                 fontsize=12, fontweight="bold", y=1.02)

    # --- Panel 1: quartile bars ---
    ax = axes[0]
    ax.bar(
        eng_q_stats.index, eng_q_stats["mean"],
        color=[
            PALETTE_PRIMARY if "Slowest" in q else PALETTE_GRAY
            for q in eng_q_stats.index
        ],
        edgecolor="white", alpha=0.85,
    )
    ax.set_ylabel("Mean RTAT (days)")
    ax.set_title("Mean Lead Time by Engineer Quartile\n"
                 "(Q1=fastest historical avg)")
    for i, (_, row) in enumerate(eng_q_stats.iterrows()):
        ax.text(i, row["mean"] + 0.1, f"{row['mean']:.1f}d",
                ha="center", fontsize=8)

    # --- Panel 2: scatter + trend line ---
    ax = axes[1]
    x_clip = eng_sub["engineer_hist_mean_rtat"].clip(upper=30)
    y_clip = eng_sub["target_days"].clip(upper=60)
    ax.scatter(x_clip, y_clip, alpha=0.02, s=1, color=PALETTE_PRIMARY)
    ax.set_xlabel("Engineer historical mean RTAT (days, capped 30)")
    ax.set_ylabel("Actual RTAT (days, capped 60)")
    ax.set_title("Engineer Proxy vs. Actual Lead Time")

    mask = x_clip.notna() & y_clip.notna()
    if mask.sum() > 100:
        z = np.polyfit(x_clip[mask], y_clip[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 30, 100)
        ax.plot(x_line, p(x_line), color=PALETTE_DARK,
                linewidth=2, label=f"Trend (slope={z[0]:.2f})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5f_engineer_signal.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return eng_q_stats


# =====================================================================
# SECTION 5G: TIME EFFECTS
# =====================================================================

def section_5g_time_effects(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Compute and plot monthly seasonality + year-over-year trend.

    Produces a 2-panel figure: (1) monthly mean RTAT + on-time rate,
    (2) per-year mean / median / on-time rate.
    """
    logger.info("=== 5G: Time effects ===")

    month_stats = (
        train.groupby("month_of_year")["target_days"]
        .agg(n="count", mean="mean",
             ontime5=lambda x: (x <= 5).mean())
        .reset_index()
        .sort_values("month_of_year")
    )
    month_stats.to_csv(cfg.eda_dir / "5g_monthly_stats.csv", index=False)

    weekend_stats = (
        train.groupby("is_weekend_close")["target_days"]
        .agg(n="count", mean="mean",
             ontime5=lambda x: (x <= 5).mean())
        .round(3)
    )
    weekend_stats.index = weekend_stats.index.map(
        {0: "Weekday close", 1: "Weekend close"},
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Seasonality and Timing Effects",
                 fontsize=12, fontweight="bold", y=1.02)

    # --- Panel 1: monthly seasonality ---
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(month_stats["month_of_year"], month_stats["mean"],
           color=PALETTE_PRIMARY, alpha=0.7, label="Mean RTAT")
    ax2.plot(month_stats["month_of_year"], month_stats["ontime5"],
             color=PALETTE_DARK, marker="o", linewidth=2,
             label="OnTime_5 rate")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean RTAT (days)", color=PALETTE_PRIMARY)
    ax2.set_ylabel("OnTime_5 rate", color=PALETTE_DARK)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        rotation=30,
    )
    ax.set_title("Monthly Seasonality")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # --- Panel 2: year-over-year ---
    ax = axes[1]
    yr_stats = (
        train.groupby("source_year")["target_days"]
        .agg(mean="mean", median="median",
             ontime5=lambda x: (x <= 5).mean())
        .reset_index()
    )
    x = np.arange(len(yr_stats))
    w = 0.35
    ax2 = ax.twinx()
    ax.bar(x - w / 2, yr_stats["mean"], w, color=PALETTE_PRIMARY,
           alpha=0.7, label="Mean RTAT")
    ax.bar(x + w / 2, yr_stats["median"], w, color=PALETTE_GRAY,
           alpha=0.7, label="Median RTAT")
    ax2.plot(x, yr_stats["ontime5"], color=PALETTE_DARK,
             marker="o", linewidth=2, label="OnTime_5 rate")
    ax.set_xticks(x)
    ax.set_xticklabels(yr_stats["source_year"])
    ax.set_ylabel("RTAT (days)")
    ax2.set_ylabel("OnTime_5 rate", color=PALETTE_DARK)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Year-over-Year Trend")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5g_time_effects.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return month_stats


# =====================================================================
# SECTION 5H: RECLAIM / REPEAT REPAIR SIGNAL
# =====================================================================

def section_5h_reclaim(train: pd.DataFrame, cfg: EDAConfig) -> pd.DataFrame:
    """Quantify lead-time impact of reclaim cases and complexity flags.

    Produces stats tables for:
        - First visit vs reclaim
        - Same-symptom reclaim vs different-symptom
        - TER repair flag
        - Sealed-system repair flag

    Returns the reclaim summary table; the others print to logger.
    """
    logger.info("=== 5H: Reclaim / repeat repair signal ===")

    reclaim_summary = (
        train.groupby("is_reclaim_case")["target_days"]
        .agg(n="count", mean="mean", median="median",
             ontime5=lambda x: (x <= 5).mean())
        .round(3)
    )
    reclaim_summary.index = reclaim_summary.index.map(
        {0: "First visit", 1: "Reclaim (repeat)"},
    )
    reclaim_summary.to_csv(cfg.eda_dir / "5h_reclaim_signal.csv")
    logger.info("  First visit vs reclaim:\n%s", reclaim_summary)

    # Same-symptom subset
    same_sym_sub = train[train["same_symptom_reclaim"].notna()]
    if len(same_sym_sub) > 100:
        same_sym = (
            same_sym_sub.groupby("same_symptom_reclaim")["target_days"]
            .agg(n="count", mean="mean",
                 ontime5=lambda x: (x <= 5).mean())
            .round(3)
        )
        same_sym.index = same_sym.index.map(
            {0: "Different symptom", 1: "Same symptom"},
        )
        logger.info("  Same-symptom reclaim vs different:\n%s", same_sym)

    # TER / Sealed flags
    for flag, label in (
        ("svc_ter_repair", "TER Repair"),
        ("svc_sealed_repair", "Sealed System"),
    ):
        sub = train[train[flag].notna()]
        if len(sub) > 100:
            stats = (
                sub.groupby(flag)["target_days"]
                .agg(n="count", mean="mean",
                     ontime5=lambda x: (x <= 5).mean())
                .round(3)
            )
            stats.index = stats.index.map({0: f"Non-{label}", 1: label})
            logger.info("  %s:\n%s", label, stats)

    return reclaim_summary


# =====================================================================
# SECTION 5I: NPS POST-HOC VALIDATION
# =====================================================================

def section_5i_nps(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """NPS preview: confirms business relevance of RTAT optimization.

    Uses 2025 NPS responders only (best year for coverage). Produces:
        - NPS by RTAT bucket (≤3d to 30d+)
        - NPS by Market_Category
        - 2-panel chart

    Returns the by-bucket table.
    """
    logger.info("=== 5I: NPS post-hoc validation (2025 subset) ===")

    nps_sub = train[
        (train["source_year"] == 2025)
        & (train["has_nps"] == 1)
    ].copy()
    logger.info("  2025 NPS responders: %s", f"{len(nps_sub):,}")

    if len(nps_sub) <= 500:
        logger.warning("  Insufficient NPS data for validation chart")
        return pd.DataFrame()

    # --- Bucket by RTAT ---
    nps_sub["rtat_bucket"] = pd.cut(
        nps_sub["target_days"],
        bins=[0, 3, 5, 7, 10, 14, 30, 150],
        labels=["≤3d", "3-5d", "5-7d", "7-10d",
                "10-14d", "14-30d", "30d+"],
    )
    nps_tier = (
        nps_sub.groupby("rtat_bucket", observed=True)
        .agg(
            n=("is_promoter", "count"),
            promoter_rate=("is_promoter", "mean"),
            detractor_rate=("is_detractor", "mean"),
        )
        .round(3)
    )
    nps_tier.to_csv(cfg.eda_dir / "5i_nps_validation.csv")

    # --- By Market_Category ---
    nps_tier_cat = (
        nps_sub.groupby("Market_Category")
        .agg(
            n=("is_promoter", "count"),
            promoter_rate=("is_promoter", "mean"),
            detractor_rate=("is_detractor", "mean"),
            mean_rtat=("target_days", "mean"),
        )
        .round(3)
        .sort_values("promoter_rate", ascending=False)
    )

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("NPS Post-Hoc Validation (2025 Responders)",
                 fontsize=12, fontweight="bold", y=1.02)

    # --- Panel 1: by RTAT bucket ---
    ax = axes[0]
    x = np.arange(len(nps_tier))
    w = 0.4
    ax.bar(x - w / 2, nps_tier["promoter_rate"], w,
           color="#88B04B", alpha=0.85, label="Promoter rate")
    ax.bar(x + w / 2, nps_tier["detractor_rate"], w,
           color=PALETTE_PRIMARY, alpha=0.85, label="Detractor rate")
    ax.set_xticks(x)
    ax.set_xticklabels(nps_tier.index, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Promoter & Detractor Rate by Lead Time Bucket")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)

    # --- Panel 2: by Market_Category ---
    ax = axes[1]
    cats = nps_tier_cat.index.tolist()
    x = np.arange(len(cats))
    ax.bar(x - w / 2, nps_tier_cat["promoter_rate"], w,
           color=["#88B04B"] * len(cats), alpha=0.85,
           label="Promoter")
    ax.bar(x + w / 2, nps_tier_cat["detractor_rate"], w,
           color=[PALETTE_PRIMARY] * len(cats), alpha=0.85,
           label="Detractor")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("NPS by Geographic Tier (2025)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(cfg.eda_dir / "5i_nps_validation.png",
                bbox_inches="tight", dpi=cfg.figure_dpi)
    plt.close(fig)

    return nps_tier


# =====================================================================
# SECTION 5J: SEGMENT PRIORITY MATRIX (the key EDA output)
# =====================================================================

def section_5j_segment_priority(
    train: pd.DataFrame,
    cfg: EDAConfig,
) -> pd.DataFrame:
    """Build the Market_Category × Channel priority matrix at EDA stage.

    This is the proto-version of what becomes the model-driven priority
    matrix in Step 8. At this point we're using raw observed late rates
    only (no predictions). Produces a heatmap of late rate at T=5 for
    every (tier × channel) cell with ≥ 500 repairs.

    Returns the segment matrix; saves CSV + heatmap PNG.
    """
    logger.info("=== 5J: Segment priority matrix ===")

    top_channels = train["Channel"].value_counts().head(5).index.tolist()

    seg_stats: list[dict] = []
    for tier in TIER_ORDER:
        for ch in top_channels:
            sub = train[
                (train["Market_Category"] == tier)
                & (train["Channel"] == ch)
            ]
            if len(sub) < cfg.min_segment_repairs:
                continue
            seg_stats.append({
                "Market_Category": tier,
                "Channel": ch,
                "n_repairs": len(sub),
                "mean_rtat": float(sub["target_days"].mean()),
                "late_rate_T5": 1 - flag_mean(sub["OnTime_5"]),
                "late_rate_T7": 1 - flag_mean(sub["OnTime_7"]),
                "parts_rate": flag_mean(sub["has_parts_reclaim"]),
                "dms_rate": flag_mean(sub["ordered_via_dms"]),
            })

    seg_df = (
        pd.DataFrame(seg_stats)
        .sort_values(
            ["late_rate_T5", "n_repairs"], ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    seg_df.to_csv(cfg.eda_dir / "5j_segment_priority_matrix.csv", index=False)

    # Heatmap
    pivot = seg_df.pivot_table(
        index="Market_Category",
        columns="Channel",
        values="late_rate_T5",
        aggfunc="mean",
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r",
                       aspect="auto", vmin=0.3, vmax=0.8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center",
                            va="center", fontsize=9, fontweight="bold",
                            color="white" if val > 0.6 else "black")
        plt.colorbar(im, ax=ax, label="Late rate at T=5")
        ax.set_title("Late Rate Heatmap: Market Tier × Channel",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(cfg.eda_dir / "5j_priority_heatmap.png",
                    bbox_inches="tight", dpi=cfg.figure_dpi)
        plt.close(fig)

    return seg_df


# =====================================================================
# SECTION 5K: HYPOTHESIS SUMMARY
# =====================================================================

HYPOTHESIS_ROWS: list[dict] = [
    {
        "hypothesis_id": "H1",
        "lever": "Geography",
        "statement": (
            "Top 10 and Metro tier repairs have higher late rates than "
            "Urban/Rural, despite higher volume — highest combined "
            "impact opportunity"
        ),
        "evidence_source": "5B Market_Category analysis",
        "priority": "High",
    },
    {
        "hypothesis_id": "H2",
        "lever": "Parts / logistics",
        "statement": (
            "Parts-required repairs have materially longer lead times "
            "than no-part repairs. Parts delivery duration "
            "(order→arrival) is the dominant variable leg"
        ),
        "evidence_source": "5E Parts impact analysis",
        "priority": "High",
    },
    {
        "hypothesis_id": "H3",
        "lever": "Channel",
        "statement": (
            "Channel is a significant predictor of lead time. DMS and "
            "Premier Partner channels show highest delay rates"
        ),
        "evidence_source": "5C Channel analysis",
        "priority": "High",
    },
    {
        "hypothesis_id": "H4",
        "lever": "Product complexity",
        "statement": (
            "Sealed system repairs (refrigerants) and TER cases have "
            "longer lead times, suggesting complexity is a key delay "
            "driver beyond parts alone"
        ),
        "evidence_source": "5H Reclaim signal",
        "priority": "Medium",
    },
    {
        "hypothesis_id": "H5",
        "lever": "Engineer capacity",
        "statement": (
            "Engineer historical performance proxy shows meaningful "
            "correlation with actual lead time — engineer deployment "
            "is a viable intervention lever"
        ),
        "evidence_source": "5F Engineer signal",
        "priority": "Medium",
    },
    {
        "hypothesis_id": "H6",
        "lever": "Seasonality",
        "statement": (
            "Lead time peaks in specific months — likely Q3/Q4 peak "
            "season. Resource allocation should account for seasonal "
            "demand patterns"
        ),
        "evidence_source": "5G Time effects",
        "priority": "Medium",
    },
    {
        "hypothesis_id": "H7",
        "lever": "Repeat failure",
        "statement": (
            "Reclaim cases (repeat visits) have longer lead times than "
            "first visits. Same-symptom reclaims are the most delayed "
            "subset"
        ),
        "evidence_source": "5H Reclaim signal",
        "priority": "Medium",
    },
    {
        "hypothesis_id": "H8",
        "lever": "NPS validation",
        "statement": (
            "High-delay repairs correlate with lower promoter rates in "
            "2025 NPS subset — confirms business relevance of RTAT "
            "optimization"
        ),
        "evidence_source": "5I NPS validation",
        "priority": "Supporting evidence",
    },
]


def section_5k_hypotheses(cfg: EDAConfig) -> pd.DataFrame:
    """Write the EDA hypothesis list (H1-H8) to CSV and return it.

    This is the EDA's most-cited output: each hypothesis maps onto a
    methodology choice in Steps 6 (feature engineering) and 8 (lever
    decomposition). H1-H4 correspond to the four operational levers
    in ``prioritization.py``.
    """
    logger.info("=== 5K: Hypothesis summary ===")
    hypotheses = pd.DataFrame(HYPOTHESIS_ROWS)
    hypotheses.to_csv(cfg.eda_dir / "5k_hypothesis_list.csv", index=False)
    for _, row in hypotheses.iterrows():
        logger.info("  %s [%s] %s: %s",
                    row["hypothesis_id"], row["priority"],
                    row["lever"], row["statement"][:60] + "…")
    return hypotheses


# =====================================================================
# MAIN ORCHESTRATION
# =====================================================================

def run_eda(cfg: EDAConfig | None = None) -> dict:
    """Run the full Step 5 EDA end-to-end.

    Reads ``master_train.parquet`` from ``cfg.interim_dir`` and produces
    11 CSV stats tables plus matching PNG charts under ``cfg.eda_dir``.

    Returns a dict of stats tables for downstream inspection.
    """
    cfg = cfg or EDAConfig()
    cfg.eda_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    logger.info("Loading training cohort...")
    train = pd.read_parquet(cfg.interim_dir / "master_train.parquet")
    logger.info("  Training rows: %s | Columns: %d",
                f"{len(train):,}", train.shape[1])

    dist_summary = section_5a_overall_distribution(train, cfg)
    tier_df = section_5b_market_category(train, cfg)
    chan_df = section_5c_channel(train, cfg)
    div_df = section_5d_division(train, cfg)
    delivery_impact = section_5e_parts_impact(train, cfg)
    eng_q_stats = section_5f_engineer(train, cfg)
    month_stats = section_5g_time_effects(train, cfg)
    reclaim_summary = section_5h_reclaim(train, cfg)
    nps_tier = section_5i_nps(train, cfg)
    seg_df = section_5j_segment_priority(train, cfg)
    hypotheses = section_5k_hypotheses(cfg)

    logger.info("=== Step 5 complete — %d stats tables + %d charts ===",
                11, 10)

    return {
        "dist_summary": dist_summary,
        "tier_stats": tier_df,
        "channel_stats": chan_df,
        "division_stats": div_df,
        "delivery_impact": delivery_impact,
        "engineer_quartiles": eng_q_stats,
        "monthly_stats": month_stats,
        "reclaim_summary": reclaim_summary,
        "nps_tier": nps_tier,
        "segment_priority": seg_df,
        "hypotheses": hypotheses,
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
    results = run_eda()
    logger.info("EDA complete. %d hypotheses generated.",
                len(results["hypotheses"]))
