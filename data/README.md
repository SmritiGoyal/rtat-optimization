# Data

This project operates on three independent operational tables exported from a Fortune 500 appliance manufacturer's US field service systems. The raw data is **client-confidential and is not committed to this repository**.

This README documents the expected schema so the pipeline can be run on any dataset with matching structure.

## Expected files

The pipeline looks for files in `data/raw/` whose names contain these substrings (case-insensitive):

| Pattern | Description | Expected size |
|---|---|---|
| `master_repair_data` | Multi-sheet Excel, one sheet per source year | ~1.5 GB |
| `parts_ledger` | One or more Excel files, sheets per source year | ~600 MB total |
| `reclaim_records` | One or more Excel files, sheets per source year | ~250 MB total |

Files with names like `master_repair_data_2023_2026.xlsx`, `parts_ledger_2024.xlsx`, or `reclaim_records_2025_2026.xlsx` will all match correctly. The pipeline uses substring matching so the surrounding naming convention is flexible.

Year information is extracted from **sheet names** via regex (`20\d{2}`), not from filenames. A workbook that combines multiple years should have one sheet per year (e.g., `2025 DMS Parts` and `2026 DMS Parts`).

## Master repair data schema

One row per repair order. Each sheet of the workbook represents one source year (2023, 2024, 2025, 2026). Required columns:

| Column | Type | Description |
|---|---|---|
| `Repair_No` | string | Primary repair identifier (join key) |
| `Warranty_Closed_Date` | YYYYMMDD int | Date repair was closed (parsed with explicit `format='%Y%m%d'`) |
| `Month_` | YYYYMM int | Month identifier |
| `RTAT_Numerator` | float | Repair turn-around time in days (the target) |
| `RTAT_Denominator` | float | Working-days denominator (not used in modeling) |
| `SVC_Engineer_Code` | string | Anonymized technician identifier |
| `SVC_Center_Type` | string | Service center type (Direct / Affiliate / ASC / etc.) |
| `Channel` | string | Service channel (see channel glossary below) |
| `Product3_Code` | string | Product hierarchy code (level 3) |
| `Product3_Name` | string | Product hierarchy name (level 3) |
| `Division_Name` | string | Top-level product division (Refrigerator / Washer / etc.) |
| `State_` | string | Two-letter US state code |
| `City_` | string | City name |
| `General_Market` | string | Market region |
| `Market_Category` | string | Market tier: `1. Top 10` / `2. Metro` / `3. Urban` / `4. Rural` |
| `Promoter` | 0/1 | NPS promoter flag (post-hoc validation only — never used as feature) |
| `Passive` | 0/1 | NPS passive flag (post-hoc validation only) |
| `Detractor` | 0/1 | NPS detractor flag (post-hoc validation only) |

## Parts ledger schema

One row per part order event. Multiple rows per repair are aggregated to one row per repair in feature engineering. Streaming-friendly: the pipeline reads these files row-by-row using openpyxl's read-only mode, so the multi-million-row sheets fit on a 16 GB laptop.

| Column | Type | Description |
|---|---|---|
| `Repair_Receipt_No` | string | Join key to `Repair_No` (after `clean_key` normalization) |
| `Parts_No` | string | Part identifier |
| `Order_Qty` | int | Quantity ordered |
| `Order_Timestamp` | datetime | When the part was ordered |
| `Picking_Release_Timestamp` | datetime | When warehouse released for picking |
| `Actual_Shipment_Timestamp` | datetime | When the part shipped |
| `Arrival_Date` | datetime | When the part arrived |
| `Shipping_Method` | string | Bucketed to `OVERNIGHT` / `TWO_DAY` / `GROUND` / `PICKUP` / `OTHER` / `UNKNOWN` |
| `Division_Name` | string | Product division |
| `ProdL2_Name` | string | Product hierarchy level 2 |
| `SO_Type` | string | Sales order type (kept for diagnostics — 100% constant, excluded from model) |
| `SO_Line_Type` | string | Sales order line type |

The 2024 and 2025 parts sheets typically hit Excel's 1,048,576-row limit. The pipeline detects truncation via `sheet.max_row == 1048576` and produces a `parts_truncation_flag` quality column.

## Reclaim records schema

One row per callback / repeat repair event. Provides the ground-truth parts-required signal and several operational flags. Read with `pd.read_excel` since the sheets are smaller (~150K-700K rows per year).

| Column | Type | Description |
|---|---|---|
| `GSFS_Repair_Header_No` | string | Join key (after normalization) |
| `Warranty_Flag` | string | Warranty status |
| `SVC_Center_Type` | string | Service center type |
| `SVC_Engineer_Code` | string | Engineer code |
| `Ship_To_Code` | string | Shipping destination code |
| `Primary_Defect_Code` | string | Top-level defect classification |
| `Primary_Defect_Desc` | string | Defect description |
| `Primary_Repair_Code` | string | Repair action code |
| `Primary_Repair_Desc` | string | Repair action description |
| `Receipt_Symptom` | string | Symptom reported at intake |
| `SVC_Symptom` | string | Symptom recorded at service |
| `Repair_Receipt_Timestamp` | datetime | Repair receipt time |
| `Repair_End_Timestamp` | datetime | Repair completion time |
| `Parts_No1` … `Parts_No5` | string | Up to 5 parts referenced (used to derive `has_parts_reclaim`, `parts_count_reclaim`) |
| `Parts_Desc1` … `Parts_Desc5` | string | Part descriptions |
| `Reclaim_Period` | float | Days since prior visit (for repeat repairs) |
| `Same_Symptom_Reclaim` | flag | Repeat visit with same symptom |
| `Same_Servicer_Flag` | flag | Repeat visit with same engineer |
| `Same_Day_Dispatch_Flag` | flag | Same-day dispatch indicator |
| `Reversed_Case_Flag` | flag | Case reversal indicator |
| `SVC_Sealed_Repair` | flag | Sealed-system repair (refrigerant, compressor) |
| `SVC_TER_Repair` | flag | TER (expedited) repair flag |
| `SVC_Part_Usage` | flag | Whether parts were used |
| `Reclaim_Symptom` | string | Symptom at reclaim |
| `Division_Code` | string | Product division code |
| `Division_Name` | string | Product division name |
| `Product2_Name` | string | Product hierarchy level 2 |
| `Product3_Name` | string | Product hierarchy level 3 |
| `Model_Code` | string | Model code |

The flag columns arrive in heterogeneous formats (`Y` / `Yes` / `1` / `TRUE` / `Sealed Repair` / `X` for positive; `N` / `No` / `0` / `FALSE` for negative; null sentinels for missing). The pipeline's `normalize_flag()` function parses all variants to nullable `Int8`.

## Service channel glossary

The `Channel` column in master data contains seven values. The `channel_risk_ordinal` encoding (1-7) is ordered by **late rate at T=5** in training data, which differs slightly from ordering by mean RTAT — Premier Partner has a lower mean RTAT than ASC but a higher T=5 late rate, so its risk ordinal sits above ASC.

| Channel | Type | Approx. mean RTAT | Risk ordinal |
|---|---|---:|---:|
| `DMS` | Direct — in-house technician network | ~6.0 days | 1 |
| `DMS2` | Direct — secondary in-house network | ~7.2 days | 2 |
| `ASC` | Authorized Service Center | ~10.2 days | 3 |
| `Premier Partner` | High-volume authorized third-party partner | ~9.7 days | 4 |
| `ASD` | Authorized Service Distributor (regional) | ~15.3 days | 5 |
| `AE` | Authorized Engineer (individual credentialed) | ~19.9 days | 6 |
| `SPO` | Service Partner Other (residual) | ~20.8 days | 7 |

The pipeline encodes these to `channel_risk_ordinal` 1-7 in the risk-ordinal order above (by T=5 late rate). Note Premier Partner's ~9.7-day mean sits just below ASC's ~10.2 despite its higher risk ordinal — the two orderings differ because the ordinal is by late rate, not mean. Unknown channels fall back to ordinal 3 (ASC-level risk).

## Cohort filter

The pipeline applies the following inclusion rules at the cohort stage (Step 4C of `ingestion.py`):

1. `repair_no_clean` must be non-null after key normalization
2. `RTAT_Numerator` must parse as numeric
3. RTAT must be ≥ 0 and ≤ 365 days
4. `Division_Name` must not be in the non-appliance list (HANDSET, LED Signage, Signage, Commercial TV, MNT Signage, PTV, Robot Business Task)
5. `SVC_Center_Type` must not be in the non-RTAT list (Affiliate, DSC, COMMERCIAL ASC (Non Referral))

The 2023-2025 records form the training + validation cohort; 2026 records are held out as the strict out-of-time test set.

Of the **2,192,254 total master rows**, the cohort filter retains **1,640,829** (74.8%): 1,060,649 for training (2023-2024), 509,930 for validation (2025), and 70,250 for holdout (2026).

## Reproducing the pipeline

The pipeline runs as six independent stages, one per file. Each stage reads the prior stage's artifacts from `outputs/` and produces its own.

```bash
# 1. Place the Excel files in data/raw/ with names matching the patterns above
mkdir -p data/raw
# (copy your master, parts, reclaim files into data/raw/)

# 2. Run the full pipeline (one stage at a time)
python ingestion.py             # Steps 1-4: master + parts + reclaim → integrated parquet
python eda.py                   # Step 5: first-pass EDA, hypothesis list, charts
python feature_engineering.py   # Step 6: feature build (40 numeric + 1 categorical), leakage review, data dictionary
python modeling.py              # Step 7: 7 classifiers + 8 regressors, threshold sweep
python prioritization.py        # Step 8: priority matrix, 4-lever decomposition
python leakage_audit.py         # Step 9: 5-test feature audit
```

All artifacts are written under `outputs/`:
- `outputs/interim/` — Step 1-4 parquets and CSVs
- `outputs/eda/` — 11 stats tables + 10 PNG charts
- `outputs/features/` — feature_train.parquet, feature_holdout.parquet, leakage review
- `outputs/models/` — LightGBM/XGBoost model pickles + result CSVs + leakage audit
- `outputs/prioritization/` — priority matrix, lever decomposition, NPS validation

All intermediate artifacts are gitignored. See [`outputs/README.md`](../outputs/README.md) for the artifact catalog.

## Access constraints

This pipeline was developed against client-confidential data. The repository contains no actual data files. Sample synthetic data matching this schema would be needed to test the pipeline end-to-end on a fresh clone.
