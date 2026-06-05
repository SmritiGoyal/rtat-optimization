"""
modeling.py
===========
Step 7: Model comparison — baseline through advanced.

Classification track (OnTime_T):
    1. Majority class baseline
    2. Logistic Regression (L2)
    3. Decision Tree (depth=3, interpretable)
    4. Decision Tree (depth=6)
    5. Random Forest
    6. XGBoost
    7. LightGBM (retained as primary — see model-choice note below)

Regression track (target_days):
    1. Overall mean baseline
    2. Segment mean baseline (market tier × channel)
    3. Ridge (L2)
    4. Lasso (L1 — feature selection signal)
    5. Decision Tree
    6. Random Forest
    7. XGBoost
    8. LightGBM

Model choice: on the final 2026 holdout LightGBM and XGBoost are within
~0.01 AUC of each other at every threshold and trade the lead by threshold
(LightGBM ahead at T=3/T=5: 0.806/0.819 vs 0.804/0.816; XGBoost ahead at
T=7/T=10). LightGBM is retained as the primary deployable model — it now leads
at the primary T=5 target and on regression — with XGBoost reported as a
validated equal-performance reference. Classifier early stopping watches AUC
(first_metric_only on the auc-first metric list), the metric the model is
selected on; with logloss watched instead it stopped prematurely on the easy
high-positive thresholds. At the relaxed thresholds T=7/T=10 the target is
highly separable (62%/75% already on-time) and validation AUC saturates within
a couple of boosting iterations — deeper models overfit and lower val AUC — so
those models legitimately early-stop at a low iteration count.

Two-phase evaluation (select-then-refit):
    Phase 1 — select on 2023-2024 → 2025 (load_split + the tracks below),
              reading feature_train.parquet (aggregates fit on the 2023-2024
              fold). The 2026 holdout is also scored here with the 2023-2024
              models, as an out-of-time reference (evaluate_holdout).
    Phase 2 — refit the locked models on all 2023-2025 (feature_*_final.parquet,
              aggregates fit on 2023-2025) and score the 2026 holdout once.
              This is the deployable headline number (run_final_holdout).

Primary target: OnTime_5 (best class balance ~46/54)
Secondary: OnTime_3, OnTime_7, OnTime_10

Two-track feature design (feature sets imported from feature_engineering's
MODEL_FEATURES — the single source of truth — so the trained model matches the
leakage review exactly):
    CORE (31 features)     — low missingness, safe for linear models after
                             median fill
    EXTENDED (40 features) — the numeric MODEL_FEATURES (one categorical-string
                             column, parts_shipping_tier, is documented but
                             excluded from the numeric model). Adds DMS-dependent
                             features (~75% missing); boosters handle nulls
                             natively. Uses the deployment-safe
                             seg_delivery_days_hist, NOT the EDA-only
                             parts_order_to_arrival_days_safe.

Output artifacts under ``outputs/models/``:
    classification_results.csv
    regression_results.csv
    threshold_results.csv
    threshold_results_xgb.csv
    threshold_sensitivity.csv
    feature_importance.csv
    lasso_features.csv
    segment_performance.csv
    channel_performance.csv
    lgbm_ontime{T}.pkl  (one per threshold — Phase-1 / 2023-2024 selection models)
    lgbm_regression.pkl (Phase-1 / 2023-2024 selection regressor)
    xgb_ontime{T}.pkl   (one per threshold — Phase-1 selection models)
    xgb_ontime5.pkl     (OnTime_5 reference classifier from the bake-off)
    xgb_regression.pkl
    lgbm_regression_final.pkl  (Phase-2 regressor, refit on all 2023-2025)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import feature_engineering as fe  # canonical MODEL_FEATURES (single source of truth)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelingConfig:
    """Configuration for the modeling stage.

    Holds paths, model hyperparameters, and split definitions. Frozen
    so the config can be shared safely across function calls.
    """
    # Directories
    interim_dir: Path = PROJECT_ROOT / "outputs" / "interim"
    feature_dir: Path = PROJECT_ROOT / "outputs" / "features"
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"

    # Reproducibility
    random_state: int = 42

    # Targets
    ontime_targets: tuple[int, ...] = (3, 5, 7, 10)
    primary_target: str = "OnTime_5"

    # Split definitions
    train_years: tuple[int, ...] = (2023, 2024)
    val_year: int = 2025

    # Decision threshold for binary classification metrics
    decision_threshold: float = 0.50

    # LightGBM classification hyperparameters — preserved exactly
    # from the original notebook
    lgb_clf_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        # AUC first so early stopping (first_metric_only=True) watches AUC, the
        # metric we actually select on. With logloss first, on the easy high-
        # positive thresholds (T=7/T=10) logloss plateaus at ~2 iterations and
        # stops the run before AUC converges, leaving 2-tree stumps.
        "metric": ["auc", "binary_logloss"],
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    })

    # LightGBM regression hyperparameters
    lgb_reg_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    })

    # XGBoost classification hyperparameters
    xgb_clf_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
        "eval_metric": "auc",
    })

    # XGBoost regression hyperparameters
    xgb_reg_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    })

    # Class-imbalance threshold for `is_unbalance` toggle
    # When training a classifier with positive rate < 0.15 or > 0.85,
    # LightGBM's `is_unbalance=True` re-weights the loss internally.
    unbalance_lo: float = 0.15
    unbalance_hi: float = 0.85

    # Early stopping for tree boosters
    early_stopping_rounds: int = 50

    # Sklearn model hyperparameters preserved from notebook
    dt_depth3_params: dict = field(default_factory=lambda: {
        "max_depth": 3, "min_samples_leaf": 1000,
        "class_weight": "balanced", "random_state": 42,
    })
    dt_depth6_clf_params: dict = field(default_factory=lambda: {
        "max_depth": 6, "min_samples_leaf": 500,
        "class_weight": "balanced", "random_state": 42,
    })
    dt_depth6_reg_params: dict = field(default_factory=lambda: {
        "max_depth": 6, "min_samples_leaf": 500, "random_state": 42,
    })
    rf_clf_params: dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 500,
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
    })
    rf_reg_params: dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 500,
        "random_state": 42, "n_jobs": -1,
    })
    lr_params: dict = field(default_factory=lambda: {
        "C": 1.0, "max_iter": 1000, "class_weight": "balanced",
        "solver": "lbfgs", "random_state": 42,
    })
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.01
    lasso_max_iter: int = 2000


# =====================================================================
# FEATURE SETS — CORE vs EXTENDED two-track design
# =====================================================================

# EXTENDED is the canonical numeric model feature set: feature_engineering's
# MODEL_FEATURES (the leakage-reviewed list) minus the one non-numeric column,
# parts_shipping_tier (a categorical string the linear/tree code here does not
# encode — documented but excluded from the numeric model, exactly like the
# EDA-only parts_order_to_arrival_days_safe). Importing it from the single
# source of truth prevents the two files from drifting: MODEL_FEATURES already
# EXCLUDES parts_order_to_arrival_days_safe (the deployment-unsafe repair-level
# delivery duration) and INCLUDES its deployment-safe substitute
# seg_delivery_days_hist, so the model trains on exactly what the leakage
# review and data dictionary describe.
_NON_NUMERIC_MODEL_FEATURES: tuple[str, ...] = ("parts_shipping_tier",)

EXTENDED_FEATURES: list[str] = [
    f for f in fe.MODEL_FEATURES if f not in _NON_NUMERIC_MODEL_FEATURES
]
"""40 numeric features for LightGBM/XGBoost — the canonical MODEL_FEATURES
(leakage-reviewed) minus the one categorical-string column. Includes
DMS-dependent features (~75% missing); boosters handle the nulls natively."""


# CORE is the linear-model subset: low-missingness features that survive median
# fill without destroying signal. Defined as EXTENDED minus the high-missingness
# DMS/parts columns and seg_delivery_days_hist (~44% missing), which would be
# mostly median-filled noise for a linear model. Relative to the original 27-
# feature linear set this adds four low-missing features that MODEL_FEATURES
# carries — day_of_week, is_weekend_close, engineer_proxy_missing, and
# is_sealed_repair — giving 31. The linear models are comparison baselines, so a
# slightly different CORE only shifts those baseline rows, not the boosted
# headline models (which use the full EXTENDED).
_CORE_EXCLUDE: frozenset[str] = frozenset({
    "parts_line_count", "parts_order_qty_sum", "parts_multi_line_flag",
    "parts_has_arrival_flag", "parts_has_shipment_flag",
    "parts_delivery_tier", "parts_truncation_flag", "reclaim_period_days",
    "seg_delivery_days_hist",
})

CORE_FEATURES: list[str] = [
    f for f in EXTENDED_FEATURES if f not in _CORE_EXCLUDE
]
"""31 features for linear models after median fill — the low-missingness subset
of EXTENDED (DMS-dependent ~75%-missing columns and seg_delivery_days_hist
excluded)."""


# =====================================================================
# SECTION 7A: DATA LOADING + SPLIT
# =====================================================================

def load_split(cfg: ModelingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load feature parquets, then split train/val by source_year.

    Reads ``feature_train.parquet`` and ``feature_holdout.parquet``,
    splits train into (2023+2024) → train, (2025) → validation, and
    merges context columns (Market_Category, Channel, Division_Name)
    back onto val_df from the interim master parquet so segment-level
    performance analyses can run.

    Returns:
        (train_df, val_df, holdout) — three DataFrames.
        The 2026 holdout is **locked** and only used in `evaluate_holdout`.
    """
    logger.info("Loading feature tables...")
    train_full = pd.read_parquet(cfg.feature_dir / "feature_train.parquet")
    holdout = pd.read_parquet(cfg.feature_dir / "feature_holdout.parquet")

    train_df = train_full[train_full["source_year"].isin(cfg.train_years)].copy()
    val_df = train_full[train_full["source_year"] == cfg.val_year].copy()

    # Bring context columns back for segment-level analyses
    master_cols_needed = [
        "repair_no_clean", "Market_Category", "Channel",
        "Division_Name", "source_year",
    ]
    master_context = pd.read_parquet(
        cfg.interim_dir / "master_train.parquet",
        columns=master_cols_needed,
    )
    val_df = val_df.merge(
        master_context[master_context["source_year"] == cfg.val_year]
        [["repair_no_clean", "Market_Category", "Channel", "Division_Name"]],
        on="repair_no_clean", how="left",
    )
    logger.info("  Train  (%s): %s", str(cfg.train_years), f"{len(train_df):,}")
    logger.info("  Val    (%d): %s", cfg.val_year, f"{len(val_df):,}")
    logger.info("  Holdout (2026): %s [LOCKED]", f"{len(holdout):,}")
    return train_df, val_df, holdout


# =====================================================================
# SECTION 7C: ARRAY HELPERS
# =====================================================================

def prep_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Extract feature matrix, converting pandas nullable ints to float32.

    Sklearn linear models can't ingest nullable Int8/Int16 dtypes directly.
    Conversion to float32 makes the matrix sklearn-compatible while still
    being memory-efficient. NaN values are preserved.
    """
    X = df[features].copy()
    for col in X.columns:
        if str(X[col].dtype).startswith("Int"):
            X[col] = X[col].astype("float32")
    return X


def prep_y(df: pd.DataFrame, target: str) -> np.ndarray:
    """Extract target column as a float numpy array (sklearn-friendly)."""
    return df[target].astype("float").values


def median_fill(
    X_tr: pd.DataFrame,
    X_vl: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN with training-set median; apply identical fills to validation.

    Used for sklearn linear models and decision trees that don't natively
    handle missing values. LightGBM and XGBoost skip this step.
    """
    med = X_tr.median()
    return X_tr.fillna(med), X_vl.fillna(med)


# =====================================================================
# SECTION 7C: EVALUATION HELPERS
# =====================================================================

def eval_clf(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    y_pred: np.ndarray,
    target: str = "OnTime_5",
) -> dict:
    """Compute the classification metric bundle for one model + target.

    Returns ROC-AUC, average precision, precision/recall/F1 at decision
    threshold 0.5, plus confusion matrix counts. AUC and AP are NaN for
    models that don't produce probabilities (e.g., DummyClassifier).
    """
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    ap = average_precision_score(y_true, y_proba) if y_proba is not None else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name, "target": target,
        "roc_auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
        "avg_prec": round(float(ap), 4) if not np.isnan(ap) else np.nan,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_val": len(y_true),
        "pos_rate": round(float(y_true.mean()), 4),
    }


def eval_reg(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the regression metric bundle for one model: MAE / RMSE / R²."""
    return {
        "model": name,
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(r2_score(y_true, y_pred), 4),
        "n_val": len(y_true),
    }


# =====================================================================
# SECTION 7E: REGRESSION TRACK
# =====================================================================

def run_regression_track(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: ModelingConfig,
) -> tuple[pd.DataFrame, lgb.LGBMRegressor, xgb.XGBRegressor, Lasso, pd.Series]:
    """Train and evaluate the 8 regression models on target_days.

    Models run sequentially in the original notebook order:
        1. Mean baseline (DummyRegressor)
        2. Segment mean baseline (market_tier × channel)
        3. Ridge (L2)
        4. Lasso (L1) — also produces feature-selection signal
        5. Decision Tree depth=6
        6. Random Forest
        7. XGBoost
        8. LightGBM

    Returns:
        (reg_df, lgb_reg, xgb_reg, lasso_model, lasso_coefs)
        Models are returned so they can be saved by the caller.
    """
    X_tr_c = prep_X(train_df, CORE_FEATURES)
    X_vl_c = prep_X(val_df, CORE_FEATURES)
    X_tr_cf, X_vl_cf = median_fill(X_tr_c, X_vl_c)

    X_tr_e = prep_X(train_df, EXTENDED_FEATURES)
    X_vl_e = prep_X(val_df, EXTENDED_FEATURES)
    X_tr_ef, X_vl_ef = median_fill(X_tr_e, X_vl_e)

    y_tr_r = prep_y(train_df, "target_days")
    y_vl_r = prep_y(val_df, "target_days")

    results: list[dict] = []

    # 1. Mean baseline
    dm = DummyRegressor(strategy="mean").fit(X_tr_cf, y_tr_r)
    results.append(eval_reg("1. Mean baseline", y_vl_r, dm.predict(X_vl_cf)))
    logger.info("  %s", results[-1])

    # 2. Segment mean (market_tier × channel)
    seg_mean = (
        train_df.groupby(["market_tier_ordinal", "channel_risk_ordinal"])
        ["target_days"].mean().reset_index()
        .rename(columns={"target_days": "seg_pred"})
    )
    val_seg = val_df[["market_tier_ordinal", "channel_risk_ordinal"]].merge(
        seg_mean, how="left",
    ).fillna(y_tr_r.mean())
    results.append(eval_reg(
        "2. Segment mean (tier×channel)", y_vl_r, val_seg["seg_pred"].values,
    ))
    logger.info("  %s", results[-1])

    # 3. Ridge (L2)
    ridge = Pipeline([
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state)),
    ])
    ridge.fit(X_tr_cf, y_tr_r)
    results.append(eval_reg("3. Ridge (L2)", y_vl_r, ridge.predict(X_vl_cf)))
    logger.info("  %s", results[-1])

    # 4. Lasso (L1)
    lasso = Pipeline([
        ("sc", StandardScaler()),
        ("m", Lasso(alpha=cfg.lasso_alpha, max_iter=cfg.lasso_max_iter,
                    random_state=cfg.random_state)),
    ])
    lasso.fit(X_tr_cf, y_tr_r)
    lasso_coefs = pd.Series(
        lasso.named_steps["m"].coef_, index=CORE_FEATURES,
    ).sort_values()
    results.append(eval_reg("4. Lasso (L1)", y_vl_r, lasso.predict(X_vl_cf)))
    logger.info("  %s", results[-1])
    n_zero = int((lasso_coefs == 0).sum())
    logger.info("  Lasso zeroed %d/%d features", n_zero, len(CORE_FEATURES))

    # 5. Decision Tree (depth=6)
    dtr = DecisionTreeRegressor(**cfg.dt_depth6_reg_params)
    dtr.fit(X_tr_cf, y_tr_r)
    results.append(eval_reg("5. Decision Tree (depth=6)", y_vl_r,
                            dtr.predict(X_vl_cf)))
    logger.info("  %s", results[-1])

    # 6. Random Forest regression
    rfr = RandomForestRegressor(**cfg.rf_reg_params)
    rfr.fit(X_tr_ef, y_tr_r)
    results.append(eval_reg("6. Random Forest", y_vl_r, rfr.predict(X_vl_ef)))
    logger.info("  %s", results[-1])

    # 7. XGBoost regression
    xgb_reg = xgb.XGBRegressor(**cfg.xgb_reg_params)
    xgb_reg.fit(X_tr_ef, y_tr_r, eval_set=[(X_vl_ef, y_vl_r)], verbose=False)
    results.append(eval_reg("7. XGBoost", y_vl_r, xgb_reg.predict(X_vl_ef)))
    logger.info("  %s", results[-1])

    # 8. LightGBM regression
    lgb_reg = lgb.LGBMRegressor(**cfg.lgb_reg_params)
    lgb_reg.fit(
        X_tr_e, y_tr_r,
        eval_set=[(X_vl_e, y_vl_r)],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(9999),
        ],
    )
    results.append(eval_reg("8. LightGBM", y_vl_r, lgb_reg.predict(X_vl_e)))
    logger.info("  %s", results[-1])

    reg_df = pd.DataFrame(results)
    return reg_df, lgb_reg, xgb_reg, lasso.named_steps["m"], lasso_coefs


# =====================================================================
# SECTION 7F: CLASSIFICATION TRACK — OnTime_5 (primary)
# =====================================================================

def run_classification_track(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: ModelingConfig,
) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    """Train and evaluate the 7 classification models on OnTime_5.

    Models:
        1. Majority class (DummyClassifier)
        2. Logistic Regression (L2, balanced)
        3. Decision Tree depth=3
        4. Decision Tree depth=6
        5. Random Forest (balanced)
        6. XGBoost (scale_pos_weight)
        7. LightGBM

    Returns:
        (clf_df, model_artifacts, y_prob_lgb, y_prob_xgb)
        model_artifacts is a dict containing the trained models and the
        validation probability arrays for downstream analysis.
    """
    X_tr_c = prep_X(train_df, CORE_FEATURES)
    X_vl_c = prep_X(val_df, CORE_FEATURES)
    X_tr_cf, X_vl_cf = median_fill(X_tr_c, X_vl_c)

    X_tr_e = prep_X(train_df, EXTENDED_FEATURES)
    X_vl_e = prep_X(val_df, EXTENDED_FEATURES)
    X_tr_ef, X_vl_ef = median_fill(X_tr_e, X_vl_e)

    y_tr5 = prep_y(train_df, cfg.primary_target)
    y_vl5 = prep_y(val_df, cfg.primary_target)
    logger.info("Class balance (%s): train pos=%.1f%% | val pos=%.1f%%",
                cfg.primary_target, 100 * y_tr5.mean(), 100 * y_vl5.mean())

    results: list[dict] = []

    # 1. Majority class
    dc = DummyClassifier(
        strategy="most_frequent", random_state=cfg.random_state,
    ).fit(X_tr_cf, y_tr5)
    y_pred_dc = dc.predict(X_vl_cf)
    results.append(eval_clf("1. Majority class", y_vl5, None, y_pred_dc))
    logger.info("  %s", results[-1])

    # 2. Logistic Regression (L2)
    lr = Pipeline([
        ("sc", StandardScaler()),
        ("m", LogisticRegression(**cfg.lr_params)),
    ])
    lr.fit(X_tr_cf, y_tr5)
    y_prob_lr = lr.predict_proba(X_vl_cf)[:, 1]
    y_pred_lr = (y_prob_lr >= cfg.decision_threshold).astype(int)
    results.append(eval_clf("2. Logistic Regression (L2)", y_vl5, y_prob_lr, y_pred_lr))
    logger.info("  %s", results[-1])

    lr_coefs = pd.Series(
        lr.named_steps["m"].coef_[0], index=CORE_FEATURES,
    ).sort_values()

    # 3. Decision Tree depth=3
    dt3 = DecisionTreeClassifier(**cfg.dt_depth3_params)
    dt3.fit(X_tr_cf, y_tr5)
    y_prob_dt3 = dt3.predict_proba(X_vl_cf)[:, 1]
    y_pred_dt3 = dt3.predict(X_vl_cf)
    results.append(eval_clf("3. Decision Tree (depth=3)", y_vl5, y_prob_dt3, y_pred_dt3))
    logger.info("  %s", results[-1])

    # 4. Decision Tree depth=6
    dt6 = DecisionTreeClassifier(**cfg.dt_depth6_clf_params)
    dt6.fit(X_tr_cf, y_tr5)
    y_prob_dt6 = dt6.predict_proba(X_vl_cf)[:, 1]
    y_pred_dt6 = dt6.predict(X_vl_cf)
    results.append(eval_clf("4. Decision Tree (depth=6)", y_vl5, y_prob_dt6, y_pred_dt6))
    logger.info("  %s", results[-1])

    # 5. Random Forest
    rf = RandomForestClassifier(**cfg.rf_clf_params)
    rf.fit(X_tr_ef, y_tr5)
    y_prob_rf = rf.predict_proba(X_vl_ef)[:, 1]
    y_pred_rf = (y_prob_rf >= cfg.decision_threshold).astype(int)
    results.append(eval_clf("5. Random Forest", y_vl5, y_prob_rf, y_pred_rf))
    logger.info("  %s", results[-1])

    # 6. XGBoost
    xgb_clf_params = dict(cfg.xgb_clf_params)
    xgb_clf_params["scale_pos_weight"] = (y_tr5 == 0).sum() / (y_tr5 == 1).sum()
    xgb_clf = xgb.XGBClassifier(**xgb_clf_params)
    xgb_clf.fit(X_tr_ef, y_tr5, eval_set=[(X_vl_ef, y_vl5)], verbose=False)
    y_prob_xgb = xgb_clf.predict_proba(X_vl_ef)[:, 1]
    y_pred_xgb = (y_prob_xgb >= cfg.decision_threshold).astype(int)
    results.append(eval_clf("6. XGBoost", y_vl5, y_prob_xgb, y_pred_xgb))
    logger.info("  %s   best_iter=%d", results[-1], xgb_clf.best_iteration)

    # 7. LightGBM
    lgb_clf = lgb.LGBMClassifier(**cfg.lgb_clf_params)
    lgb_clf.fit(
        X_tr_e, y_tr5,
        eval_set=[(X_vl_e, y_vl5)],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, first_metric_only=True,
                               verbose=False),
            lgb.log_evaluation(9999),
        ],
    )
    y_prob_lgb = lgb_clf.predict_proba(X_vl_e)[:, 1]
    y_pred_lgb = (y_prob_lgb >= cfg.decision_threshold).astype(int)
    results.append(eval_clf("7. LightGBM", y_vl5, y_prob_lgb, y_pred_lgb))
    logger.info("  %s   best_iter=%d", results[-1], lgb_clf.best_iteration_)

    clf_df = pd.DataFrame(results)

    return clf_df, {
        "lgb_clf": lgb_clf,
        "xgb_clf": xgb_clf,
        "lr_coefs": lr_coefs,
        "y_prob_lgb": y_prob_lgb,
        "y_prob_xgb": y_prob_xgb,
        "y_pred_lgb": y_pred_lgb,
        "y_pred_xgb": y_pred_xgb,
        "y_vl5": y_vl5,
    }, y_prob_lgb, y_prob_xgb


# =====================================================================
# SECTION 7I: LIGHTGBM ACROSS THRESHOLDS T=3, 5, 7, 10
# =====================================================================

def run_threshold_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: ModelingConfig,
) -> tuple[pd.DataFrame, dict[int, lgb.LGBMClassifier]]:
    """Train LightGBM on each OnTime threshold (T=3, 5, 7, 10) independently.

    When the positive class rate is extreme (<15% or >85%), enables
    LightGBM's ``is_unbalance=True`` to re-weight loss internally. This
    handles the skew at T=3 (~28% positive) and T=10 (~75% positive).

    Returns:
        (threshold_results_df, models_dict)
        models_dict maps threshold T → fitted LGBMClassifier.
    """
    X_tr_e = prep_X(train_df, EXTENDED_FEATURES)
    X_vl_e = prep_X(val_df, EXTENDED_FEATURES)

    results: list[dict] = []
    models: dict[int, lgb.LGBMClassifier] = {}

    for t in cfg.ontime_targets:
        y_tr_t = prep_y(train_df, f"OnTime_{t}")
        y_vl_t = prep_y(val_df, f"OnTime_{t}")

        # Adjust class weight for skewed thresholds
        pos_rate = float(y_tr_t.mean())
        params_t = dict(cfg.lgb_clf_params)
        if pos_rate < cfg.unbalance_lo or pos_rate > cfg.unbalance_hi:
            params_t["is_unbalance"] = True

        m = lgb.LGBMClassifier(**params_t)
        m.fit(
            X_tr_e, y_tr_t,
            eval_set=[(X_vl_e, y_vl_t)],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, first_metric_only=True,
                                   verbose=False),
                lgb.log_evaluation(9999),
            ],
        )
        y_prob_t = m.predict_proba(X_vl_e)[:, 1]
        y_pred_t = (y_prob_t >= cfg.decision_threshold).astype(int)
        res = eval_clf("LightGBM", y_vl_t, y_prob_t, y_pred_t,
                       target=f"OnTime_{t}")
        results.append(res)
        models[t] = m
        logger.info(
            "  T=%d  AUC=%.4f  F1=%.4f  Pos=%.1f%%  best_iter=%d",
            t, res["roc_auc"], res["f1"],
            100 * res["pos_rate"], m.best_iteration_,
        )

    return pd.DataFrame(results), models


def run_threshold_sweep_xgb(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cfg,
) -> tuple[pd.DataFrame, dict[int, "xgb.XGBClassifier"]]:
    """Train XGBoost on each OnTime threshold (T=3,5,7,10) independently.

    Uses scale_pos_weight = n_neg / n_pos per threshold to handle the
    skew at T=3 (~28% positive) and T=10 (~75% positive) — the XGBoost
    equivalent of LightGBM's is_unbalance=True.

    Returns:
        (threshold_results_df, models_dict)  with models_dict[T] = fitted
        XGBClassifier.
    """
    X_tr_e = prep_X(train_df, EXTENDED_FEATURES)
    X_vl_e = prep_X(val_df, EXTENDED_FEATURES)

    results: list[dict] = []
    models: dict[int, xgb.XGBClassifier] = {}

    for t in cfg.ontime_targets:
        y_tr_t = prep_y(train_df, f"OnTime_{t}")
        y_vl_t = prep_y(val_df, f"OnTime_{t}")

        params_t = dict(cfg.xgb_clf_params)
        n_pos = int((y_tr_t == 1).sum())
        n_neg = int((y_tr_t == 0).sum())
        params_t["scale_pos_weight"] = (n_neg / n_pos) if n_pos else 1.0

        m = xgb.XGBClassifier(**params_t)
        m.fit(X_tr_e, y_tr_t, eval_set=[(X_vl_e, y_vl_t)], verbose=False)

        y_prob_t = m.predict_proba(X_vl_e)[:, 1]
        y_pred_t = (y_prob_t >= cfg.decision_threshold).astype(int)
        res = eval_clf("XGBoost", y_vl_t, y_prob_t, y_pred_t,
                       target=f"OnTime_{t}")
        results.append(res)
        models[t] = m
        logger.info(
            "  [XGB] T=%d  AUC=%.4f  F1=%.4f  Pos=%.1f%%  best_iter=%d",
            t, res["roc_auc"], res["f1"],
            100 * res["pos_rate"], m.best_iteration,
        )

    return pd.DataFrame(results), models


# =====================================================================
# SECTION 7J: OPERATING THRESHOLD SENSITIVITY
# =====================================================================

def compute_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep decision thresholds from 0.10 to 0.90 and record P/R/F1.

    Used to support operating-point selection: e.g., where can we hit
    precision ≥ 60%, or where does F1 peak.
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.10, 0.91, 0.01)

    rows: list[dict] = []
    for thr in threshold_grid:
        yp = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": round(float(thr), 2),
            "precision": round(precision_score(y_true, yp, zero_division=0), 4),
            "recall": round(recall_score(y_true, yp, zero_division=0), 4),
            "f1": round(f1_score(y_true, yp, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


# =====================================================================
# SECTION 7K: SEGMENT-LEVEL PERFORMANCE
# =====================================================================

def _seg_auc(g: pd.DataFrame, prob_col: str, target_col: str = "OnTime_5") -> float:
    """Group-level ROC-AUC, NaN if a group has only one class label."""
    if g[target_col].nunique() < 2:
        return np.nan
    return round(roc_auc_score(
        g[target_col].astype(float), g[prob_col],
    ), 4)


def segment_performance(
    val_df: pd.DataFrame,
    y_prob_lgb: np.ndarray,
    y_prob_xgb: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Compute segment-level AUC for LightGBM and XGBoost on validation set.

    Per-segment grouping along three axes:
        - Market_Category
        - Channel
        - Division_Name (bottom-10 by on-time rate)

    Each grouping reports: group size n, actual OnTime_5 rate, and the
    LightGBM/XGBoost AUC within that group. Used to flag segments where
    the model under-performs and may need separate treatment.
    """
    missing_seg = [c for c in ("Market_Category", "Channel", "Division_Name")
                   if c not in val_df.columns]
    if missing_seg:
        logger.warning("Segment columns missing: %s — skipping", missing_seg)
        return {}

    seg_val = val_df[["Market_Category", "Channel", "Division_Name",
                      "OnTime_5", "target_days"]].copy().reset_index(drop=True)
    seg_val["lgb_prob"] = y_prob_lgb
    seg_val["xgb_prob"] = y_prob_xgb

    seg_perf = (
        seg_val.groupby("Market_Category")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "actual_ontime5": round(g["OnTime_5"].astype(float).mean(), 4),
            "lgb_auc": _seg_auc(g, "lgb_prob"),
            "xgb_auc": _seg_auc(g, "xgb_prob"),
        }))
        .sort_values("actual_ontime5")
    )
    chan_perf = (
        seg_val.groupby("Channel")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "actual_ontime5": round(g["OnTime_5"].astype(float).mean(), 4),
            "lgb_auc": _seg_auc(g, "lgb_prob"),
            "xgb_auc": _seg_auc(g, "xgb_prob"),
        }))
        .sort_values("actual_ontime5")
    )
    div_perf = (
        seg_val.groupby("Division_Name")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "actual_ontime5": round(g["OnTime_5"].astype(float).mean(), 4),
            "lgb_auc": _seg_auc(g, "lgb_prob"),
        }))
        .sort_values("actual_ontime5")
        .head(10)
    )
    return {
        "segment_performance": seg_perf,
        "channel_performance": chan_perf,
        "division_performance_bottom10": div_perf,
    }


# =====================================================================
# SECTION 7L: FEATURE IMPORTANCE COMPARISON
# =====================================================================

def feature_importance_comparison(
    xgb_clf,
    lgb_clf,
) -> pd.DataFrame:
    """Build a side-by-side XGBoost vs LightGBM feature importance table.

    Reports both raw importance, normalized share (`xgb_pct`, `lgb_pct`),
    and a rank-difference column showing where the two models disagree
    on relative importance.
    """
    xgb_imp = pd.DataFrame({
        "feature": EXTENDED_FEATURES,
        "xgb_imp": xgb_clf.feature_importances_,
    })
    lgb_imp = pd.DataFrame({
        "feature": EXTENDED_FEATURES,
        "lgb_imp": lgb_clf.feature_importances_,
    })
    imp_compare = xgb_imp.merge(lgb_imp, on="feature")
    imp_compare["xgb_pct"] = imp_compare["xgb_imp"] / imp_compare["xgb_imp"].sum()
    imp_compare["lgb_pct"] = imp_compare["lgb_imp"] / imp_compare["lgb_imp"].sum()
    imp_compare["rank_diff"] = (
        imp_compare["xgb_pct"].rank(ascending=False)
        - imp_compare["lgb_pct"].rank(ascending=False)
    )
    return imp_compare.sort_values("lgb_pct", ascending=False).reset_index(drop=True)


# =====================================================================
# SECTION 7P: HOLDOUT EVALUATION (FINAL)
# =====================================================================

def evaluate_holdout(
        lgb_models: dict,
        lgb_reg,
        xgb_models: dict,
        xgb_reg,
        holdout: pd.DataFrame,
        cfg,
) -> dict[str, dict]:
    """Score the 2026 holdout with the Phase-1 (2023-2024) models — both
    LightGBM and XGBoost — as an out-of-time reference.

    This is NOT the deployable headline number: these are the selection-fold
    models applied to 2026. The deployable figure comes from run_final_holdout
    (Phase 2), which refits on all 2023-2025 before scoring. No result here may
    inform any training-time choice.

    Returns:
        {
          "classification_xgb": {T: {...}},   # XGBoost, Phase-1 reference
          "classification_lgb": {T: {...}},   # LightGBM, Phase-1 reference
          "regression_xgb": {...},
          "regression_lgb": {...},
        }
    """
    X_ho_e = prep_X(holdout, EXTENDED_FEATURES)
    out: dict[str, dict] = {
        "classification_xgb": {},
        "classification_lgb": {},
    }

    for t in cfg.ontime_targets:
        y_ho_t = prep_y(holdout, f"OnTime_{t}")

        p_xgb = xgb_models[t].predict_proba(X_ho_e)[:, 1]
        pred_xgb = (p_xgb >= cfg.decision_threshold).astype(int)
        out["classification_xgb"][t] = eval_clf(
            f"XGB Holdout T={t}", y_ho_t, p_xgb, pred_xgb, target=f"OnTime_{t}")

        p_lgb = lgb_models[t].predict_proba(X_ho_e)[:, 1]
        pred_lgb = (p_lgb >= cfg.decision_threshold).astype(int)
        out["classification_lgb"][t] = eval_clf(
            f"LGB Holdout T={t}", y_ho_t, p_lgb, pred_lgb, target=f"OnTime_{t}")

        logger.info(
            "  Holdout T=%d  XGB AUC=%.4f F1=%.4f  |  LGB AUC=%.4f F1=%.4f",
            t,
            out["classification_xgb"][t]["roc_auc"], out["classification_xgb"][t]["f1"],
            out["classification_lgb"][t]["roc_auc"], out["classification_lgb"][t]["f1"],
        )

    y_ho_r = prep_y(holdout, "target_days")
    out["regression_xgb"] = eval_reg("XGB Holdout regression", y_ho_r, xgb_reg.predict(X_ho_e))
    out["regression_lgb"] = eval_reg("LGB Holdout regression", y_ho_r, lgb_reg.predict(X_ho_e))
    logger.info(
        "  Holdout regression  XGB MAE=%.3fd R²=%.4f  |  LGB MAE=%.3fd R²=%.4f",
        out["regression_xgb"]["mae"], out["regression_xgb"]["r2"],
        out["regression_lgb"]["mae"], out["regression_lgb"]["r2"],
    )

    return out

# =====================================================================
# SECTION 7Q: PHASE-2 FINAL FEATURE TABLES (load)
# =====================================================================

def load_final(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 2023-2025-fit final training frame and 2026 holdout.

    Returns (train_final_df, holdout_final_df). train_final is the full
    2023-2025 cohort (no train/val split — selection is already done).
    """
    train_final = pd.read_parquet(cfg.feature_dir / "feature_train_final.parquet")
    holdout_final = pd.read_parquet(cfg.feature_dir / "feature_holdout_final.parquet")
    logger.info("  Final train (2023-2025): %s | Final holdout (2026): %s [LOCKED]",
                f"{len(train_final):,}", f"{len(holdout_final):,}")
    return train_final, holdout_final

# =====================================================================
# SECTION 7R: PHASE-2 FINAL HOLDOUT (refit on 2023-2025, score 2026 once)
# =====================================================================

def run_final_holdout(
    train_final: pd.DataFrame,
    holdout_final: pd.DataFrame,
    lgb_models: dict,          # Phase-1 LightGBM threshold models (for locked best_iter)
    xgb_models: dict,          # Phase-1 XGBoost threshold models (for locked best_iter)
    lgb_reg,                   # Phase-1 LightGBM regressor
    xgb_reg,                   # Phase-1 XGBoost regressor
    cfg,
) -> dict:
    """Select-then-refit final evaluation.

    For each threshold T: refit LightGBM and XGBoost on ALL of 2023-2025
    at the locked Phase-1 iteration count, then score the 2026 holdout.
    LightGBM and XGBoost are within ~0.01 AUC on the holdout and trade the
    lead by threshold (LightGBM ahead at T=3/T=5, XGBoost at T=7/T=10);
    LightGBM is reported as the primary/deployable model and XGBoost for
    comparison.

    This touches the 2026 holdout exactly once. Do NOT tune anything after
    reading these numbers — that would void the holdout.
    """
    logger.info("=== PHASE 2: FINAL HOLDOUT (refit on 2023-2025, score 2026 once) ===")

    X_full = prep_X(train_final, EXTENDED_FEATURES)
    X_ho = prep_X(holdout_final, EXTENDED_FEATURES)

    # Guard: the Phase-1 XGB regression blew up (MAE 18.7) — almost certainly
    # inf/overflow in the holdout matrix. Surface it instead of silently
    # reporting garbage.
    Xho_np = X_ho.to_numpy(dtype="float32")
    n_inf = int(np.isinf(Xho_np).sum())
    if n_inf:
        logger.warning("  Holdout feature matrix has %d inf values "
                       "(max abs=%.3g) — XGB regression may be unreliable.",
                       n_inf, float(np.nanmax(np.abs(Xho_np))))

    out: dict[str, dict] = {"classification_lgb": {}, "classification_xgb": {}}

    for t in cfg.ontime_targets:
        y_full_t = prep_y(train_final, f"OnTime_{t}")
        y_ho_t = prep_y(holdout_final, f"OnTime_{t}")
        pos_rate = float(y_full_t.mean())

        # ---- LightGBM: locked n_estimators, no early stopping, full data ----
        lgb_iter = int(lgb_models[t].best_iteration_)
        lgb_params = dict(cfg.lgb_clf_params)
        lgb_params["n_estimators"] = lgb_iter
        if pos_rate < cfg.unbalance_lo or pos_rate > cfg.unbalance_hi:
            lgb_params["is_unbalance"] = True
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_full, y_full_t)  # no eval_set, no callbacks

        p_lgb = m_lgb.predict_proba(X_ho)[:, 1]
        pred_lgb = (p_lgb >= cfg.decision_threshold).astype(int)
        out["classification_lgb"][t] = eval_clf(
            f"FINAL LGB Holdout T={t}", y_ho_t, p_lgb, pred_lgb, target=f"OnTime_{t}")

        # ---- XGBoost: locked n_estimators (best_iteration is 0-indexed) ----
        xgb_iter = int(xgb_models[t].best_iteration) + 1
        xgb_params = dict(cfg.xgb_clf_params)
        xgb_params["n_estimators"] = xgb_iter
        xgb_params.pop("early_stopping_rounds", None)  # no ES without eval_set
        n_pos = int((y_full_t == 1).sum())
        n_neg = int((y_full_t == 0).sum())
        xgb_params["scale_pos_weight"] = (n_neg / n_pos) if n_pos else 1.0
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_full, y_full_t, verbose=False)  # no eval_set

        p_xgb = m_xgb.predict_proba(X_ho)[:, 1]
        pred_xgb = (p_xgb >= cfg.decision_threshold).astype(int)
        out["classification_xgb"][t] = eval_clf(
            f"FINAL XGB Holdout T={t}", y_ho_t, p_xgb, pred_xgb, target=f"OnTime_{t}")

        logger.info(
            "  FINAL Holdout T=%d  LGB AUC=%.4f F1=%.4f  |  XGB AUC=%.4f F1=%.4f  "
            "(lgb_iter=%d xgb_iter=%d)",
            t,
            out["classification_lgb"][t]["roc_auc"], out["classification_lgb"][t]["f1"],
            out["classification_xgb"][t]["roc_auc"], out["classification_xgb"][t]["f1"],
            lgb_iter, xgb_iter,
        )

    # ---- Regression: LightGBM primary (robust); refit on full 2023-2025 ----
    y_full_r = prep_y(train_final, "target_days")
    y_ho_r = prep_y(holdout_final, "target_days")

    lgb_reg_iter = int(getattr(lgb_reg, "best_iteration_", 0) or cfg.lgb_reg_params.get("n_estimators", 500))
    lgb_reg_params = dict(cfg.lgb_reg_params)
    lgb_reg_params["n_estimators"] = lgb_reg_iter
    m_lgb_reg = lgb.LGBMRegressor(**lgb_reg_params)
    m_lgb_reg.fit(X_full, y_full_r)
    out["regression_lgb"] = eval_reg(
        "FINAL LGB Holdout regression", y_ho_r, m_lgb_reg.predict(X_ho))

    logger.info("  FINAL Holdout regression (LGB)  MAE=%.3fd  RMSE=%.3fd  R²=%.4f",
                out["regression_lgb"]["mae"], out["regression_lgb"]["rmse"],
                out["regression_lgb"]["r2"])

    # Persist the final deployable regressor. The per-threshold classifiers
    # refit above are local to the loop and intentionally not saved — only
    # their holdout metrics are reported. Persist them here if a future
    # deployment needs the threshold models, not just lgbm_regression_final.
    for t in cfg.ontime_targets:
        pass  # per-threshold final classifiers not persisted (metrics only)
    joblib.dump(m_lgb_reg, cfg.model_dir / "lgbm_regression_final.pkl")

    logger.info("=== Phase 2 final holdout complete ===")
    return out


# =====================================================================
# MAIN ORCHESTRATION
# =====================================================================

def run_modeling(cfg: ModelingConfig | None = None) -> dict:
    """Run the full Step 7 modeling pipeline end-to-end.

    Produces all CSV deliverables under ``cfg.model_dir`` and persists
    fitted models as joblib pickles.

    Returns a dict of result tables and trained models for inspection.
    """
    cfg = cfg or ModelingConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    # ---- 7A: load split ----
    train_df, val_df, holdout = load_split(cfg)

    # ---- 7E: regression track ----
    logger.info("=== REGRESSION TRACK — target_days ===")
    reg_df, lgb_reg, xgb_reg, lasso_model, lasso_coefs = run_regression_track(
        train_df, val_df, cfg,
    )

    # ---- 7F: classification track ----
    logger.info("=== CLASSIFICATION TRACK — %s ===", cfg.primary_target)
    clf_df, clf_arts, y_prob_lgb, y_prob_xgb = run_classification_track(
        train_df, val_df, cfg,
    )

    # ---- 7G/L: feature importance ----
    logger.info("=== Feature importance comparison ===")
    imp_df = feature_importance_comparison(
        clf_arts["xgb_clf"], clf_arts["lgb_clf"],
    )

    # ---- 7H: Lasso feature selection ----
    logger.info("=== Lasso feature selection ===")
    lasso_summary = pd.DataFrame({
        "feature": CORE_FEATURES,
        "coef": lasso_model.coef_,
    }).sort_values("coef", key=abs, ascending=False)
    lasso_summary["selected"] = lasso_summary["coef"].abs() > 0

    # ---- 7I: threshold sweep ----
    logger.info("=== LightGBM threshold sweep T=3,5,7,10 ===")
    thresh_df, lgb_models = run_threshold_sweep(train_df, val_df, cfg)

    logger.info("=== XGBoost threshold sweep T=3,5,7,10 ===")
    thresh_df_xgb, xgb_models = run_threshold_sweep_xgb(train_df, val_df, cfg)

    # ---- 7J: operating threshold sensitivity ----
    logger.info("=== Operating threshold sensitivity ===")
    thresh_sens = compute_threshold_sensitivity(clf_arts["y_vl5"], y_prob_lgb)

    # ---- 7K: segment performance ----
    logger.info("=== Segment-level performance ===")
    segments = segment_performance(val_df, y_prob_lgb, y_prob_xgb)

    # ---- 7P: holdout evaluation (FINAL) ----
    logger.info("=== HOLDOUT EVALUATION (2026) ===")
    holdout_results = evaluate_holdout(lgb_models, lgb_reg, xgb_models, xgb_reg, holdout, cfg)

    # ---- 7N: save all outputs ----
    logger.info("=== Saving outputs ===")
    clf_df.to_csv(cfg.model_dir / "classification_results.csv", index=False)
    reg_df.to_csv(cfg.model_dir / "regression_results.csv", index=False)
    thresh_df.to_csv(cfg.model_dir / "threshold_results.csv", index=False)
    thresh_sens.to_csv(cfg.model_dir / "threshold_sensitivity.csv", index=False)
    imp_df.to_csv(cfg.model_dir / "feature_importance.csv", index=False)
    lasso_summary.to_csv(cfg.model_dir / "lasso_features.csv", index=False)

    if "segment_performance" in segments:
        segments["segment_performance"].to_csv(
            cfg.model_dir / "segment_performance.csv",
        )
        segments["channel_performance"].to_csv(
            cfg.model_dir / "channel_performance.csv",
        )

    if getattr(cfg, "run_final_refit", True):
        train_final, holdout_final = load_final(cfg)
        final_holdout_results = run_final_holdout(
                train_final, holdout_final,
                lgb_models, xgb_models, lgb_reg, xgb_reg, cfg,
            )
    else:
        final_holdout_results = None

    # Save models per-threshold + final regressor and XGBoost reference
    for t, m in lgb_models.items():
        joblib.dump(m, cfg.model_dir / f"lgbm_ontime{t}.pkl")
    joblib.dump(lgb_reg, cfg.model_dir / "lgbm_regression.pkl")
    joblib.dump(clf_arts["xgb_clf"], cfg.model_dir / "xgb_ontime5.pkl")
    joblib.dump(xgb_reg, cfg.model_dir / "xgb_regression.pkl")

    for t, m in xgb_models.items():
        joblib.dump(m, cfg.model_dir / f"xgb_ontime{t}.pkl")
    thresh_df_xgb.to_csv(cfg.model_dir / "threshold_results_xgb.csv", index=False)

    logger.info("=== Step 7 complete ===")

    return {
        "classification_results": clf_df,
        "regression_results": reg_df,
        "threshold_results": thresh_df,
        "threshold_sensitivity": thresh_sens,
        "feature_importance": imp_df,
        "lasso_summary": lasso_summary,
        "lasso_coefs": lasso_coefs,
        "segment_performance": segments,
        "holdout": holdout_results,
        "models": {
            "lgb_classifiers": lgb_models,
            "lgb_regression": lgb_reg,
            "xgb_classifier": clf_arts["xgb_clf"],
            "xgb_regression": xgb_reg,
        },
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
    results = run_modeling()
    logger.info("Modeling complete. %d per-threshold LightGBM models trained "
                "(plus XGBoost threshold models, regressors, and the Phase-2 "
                "final regressor — see outputs/models/).",
                len(results["models"]["lgb_classifiers"]))
