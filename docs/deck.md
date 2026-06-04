# RTAT Optimization — Final Presentation

**What Really Drives Repair Turn-Around Time in Appliance Service Operations?**
*Analytics for Smarter Resource Allocation*

A capstone analytics engagement for a Fortune 500 appliance manufacturer's US field service operations team.

---

## Slide 1 — Executive Summary: Why This Project Matters

**Background & Motivation**

The client handles **over 2 million appliance repair orders each year** across its US field service network. Repair Turn-Around Time (RTAT) is a critical KPI because it directly affects both customer experience and operational efficiency.

Yet repair delays are not evenly distributed — they are concentrated in specific combinations of geography, service channel, and engineer assignment.

> *Which segments should receive resources first to achieve the greatest RTAT improvement?*

This project answers that question by identifying **where delay is concentrated, what drives it most, and which operational lever can deliver the highest impact**.

---

## Slide 2 — Understanding RTAT: What We Know and What We Don't

**What is RTAT?**

Repair Turn-Around Time — the elapsed time from repair intake to repair completion. A key KPI for both customer experience and operational efficiency.

**What we already know about RTAT variation:**

| Pattern | Observation |
|---|---|
| By product & region | RTAT differs across product categories and geographic tiers |
| By service channel | RTAT varies significantly across channels — delay is not a uniform problem |

| Fast segments | Slow segments |
|---|---|
| Urban × DMS | Urban × Premier Partner |
| Metro × DMS | Rural × Premier Partner |
| Top 10 × DMS | Rural × Authorized Engineer |

**What RTAT variation alone does NOT reveal:**
- Where to invest first
- What primarily drives delay in each segment

---

## Slide 3 — Problem Statement & Analytical Approach

**What's in our dataset?**

Repair-order-level data covering:
- Repair outcomes (RTAT, completion status)
- Service channel (7 channel types — DMS, DMS2, ASC, Premier Partner, ASD, AE, SPO)
- Geography (state, city, market tier)
- Engineer history (anonymized engineer codes)
- Parts logistics signals (ordering, shipping, arrival timestamps)
- Reclaim / repeat repair flags
- Product category and seasonality

**What we'll do with this data — three analytical tracks:**

| Track | Purpose |
|---|---|
| **Regression** | Estimate expected RTAT for each repair order using historical service data |
| **Classification** | Predict whether a repair will be completed within a target threshold (T = 3, 5, 7, 10 days) |
| **Prioritization** | Identify which segments should receive resources first to reduce RTAT most effectively |

---

## Slide 4 — Data Preparation: 41 Features, Temporal Split, 5-Test Leakage Audit

**Raw data → modeling cohort**

<!-- VERIFY: the exclusion lines below sum to 1,540,829, but the verified final
cohort (= train + val + holdout) is 1,640,829 — off by 100,000. Pull the real
per-reason counts from outputs/interim/cohort_summary.csv and correct one line
before presenting. The 1,640,829 total is correct; one exclusion figure is not. -->

```
Master Universe:                          2,192,254 repairs
  - Non-appliance divisions (2.8%):         -61,519
  - Non-RTAT center types (1.2%):           -26,938
  - Missing RTAT records (24.0%):          -525,772
  - Other exclusions:                       -37,196
--------------------------------------------------------
Final Modeling Cohort (74.8%):            1,640,829 repairs
```

**Train / Validate / Holdout split**

| Split | Source year | Rows |
|---|---|---:|
| Train | 2023 + 2024 | 1,060,649 |
| Validate | 2025 | 509,930 |
| Holdout | 2026 (Jan-Apr) | 70,250 |

A strict temporal split tests performance on truly unseen future data. Training-class aggregates are fit on the 2023-2024 fold for model selection, then refit on all 2023-2025 for the final model, which scores the 2026 holdout exactly once.

**Feature audit summary - 41 features across 8 groups**

| Group | # Features | Description | Audit Status |
|---|---:|---|---|
| Geography | 5 | Market tier, target encoding, geo aggregates | Clean |
| Channel | 3 | Channel type, risk ordinal, training aggregates | Clean |
| Time | 6 | Month, quarter, peak season indicators | Clean |
| Product | 4 | Division-level performance aggregates | Medium |
| Engineer | 3 | Historical mean RTAT proxy (fold-scoped, train fold only) | Clean |
| Reclaim | 6 | Parts flag, repair complexity, repeat-visit | Clean |
| Parts logistics | 11 | Parts ordering, logistics, segment delivery days | Conditional |
| Interaction | 3 | Geo x channel x engineer cross-features | Medium |

The leakage audit flags features that may not be deployment-safe. Note: the audit compares training against the 2026 holdout and does not, by construction, detect a leak confined to the validation fold — see Slide 7.

---

## Slide 5 — Modeling Results: Two-Level Predictive Design

**Classification Model Comparison — validation (2025), Target: T = 5 days**

| Model | AUC | Avg Precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 0.500 | 0.468 | 0.0% | 0.0% | 0.000 |
| Logistic Regression (L2) | 0.761 | 0.724 | 64.9% | 72.0% | 0.683 |
| Decision Tree (depth=3) | 0.738 | 0.657 | 62.1% | 79.3% | 0.697 |
| Decision Tree (depth=6) | 0.762 | 0.716 | 64.6% | 73.8% | 0.689 |
| Random Forest | 0.788 | 0.750 | 65.0% | 78.9% | 0.713 |
| XGBoost | 0.796 | 0.759 | 67.7% | 75.4% | 0.713 |
| LightGBM | 0.759 | 0.669 | 66.9% | 70.5% | 0.686 |

*On this single-fit validation bake-off XGBoost leads; LightGBM underfits at default iterations (best_iter ~29) because hyperparameters were not re-tuned after the leak fix. On the final 2026 holdout the two are tied — see Slide 7. LightGBM is retained as primary for regression stability and faster training.*

**Regression Model Comparison — validation (2025), MAE in days**

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Mean Baseline | 6.772 | 10.984 | -0.001 |
| Segment Mean | 5.971 | 10.191 | 0.138 |
| Ridge (L2) | 5.385 | 9.473 | 0.255 |
| Random Forest | 4.934 | 9.125 | 0.309 |
| XGBoost | 4.916 | 9.059 | 0.319 |
| LightGBM | 4.950 | 9.178 | 0.301 |

**Why LightGBM**

1. **Tied on the final holdout** — LightGBM and XGBoost land within ~0.005 AUC of each other at every threshold on the 2026 holdout (T=5: 0.807 vs 0.812). LightGBM is chosen for regression stability and ~25% faster training, not a validation-accuracy edge.
2. **Native missing-value handling** - 75% of parts-logistics features are structurally missing for repairs that don't need parts. LightGBM handles this natively via surrogate splits.
3. **Scale efficiency** - trains materially faster on 1.5M rows, enabling iterative development and rapid retraining.

> **32% MAE improvement over naive baseline** (6.77 -> 4.58 days on the 70,250-repair 2026 holdout, final model)

---

## Slide 6 — The Key Finding: What Drives Delay Most

**Top 12 Feature Importances (LightGBM)**

The most important features cluster around engineer history, channel risk, and market tier rather than parts logistics — which inverts the operational assumption the field service org had been working from.

**The ~4x engineer gap — concentrated in the slowest quartile**

RTAT varies dramatically across engineer performance quartiles for identical repair types:

| Quartile | Avg RTAT | Share of repairs |
|---|---:|---:|
| Q1 (fastest) | 4.5 days | 25.9% |
| Q2 | 6.3 days | 24.2% |
| Q3 | 8.9 days | 24.9% |
| Q4 (slowest) | **17.7 days** | 25.0% |

The Q4/Q1 ratio of **3.9x** is the #1 driver across all segments. The signal is not a smooth gradient — it's a long tail: the slowest 25% of engineers are dramatically slower than the rest. This concentration is why engineer deployment is the dominant lever (43% of segments) and why "redeploy the bottom quartile" is the most actionable single intervention.

> **Engineer deployment drives delay in 43% of segments — surpassing parts logistics (29%) as the primary lever.**

---

## Slide 7 — Evaluation: Holdout Performance, Robustness, and External Validation

**Headline holdout numbers (2026, final model — select on 2023-24 → 2025, refit on 2023-2025, score 2026 once)**

| Metric | Value |
|---|---:|
| Holdout AUC (2026, T=5) | **0.807** (XGBoost 0.812 — tied) |
| Holdout F1 (T=5) | **0.729** |
| Holdout MAE (regression) | **4.58 days** |
| Holdout R² (regression) | **0.36** |

**Model robustness benchmarks**

| Track | Holdout AUC | F1 | Leakage Status |
|---|---:|---:|---|
| 7-Feature Safe Baseline (pre-fix) | 0.650 | 0.535 | Clean |
| 19-Feature Conservative (pre-fix) | 0.768 | 0.724 | Clean |
| **Final 41-Feature LightGBM** | **0.807** | **0.729** | **Passed*** |

*Passed the 5-test leakage audit. That audit compares training against the 2026 holdout and does not detect a leak confined to the validation fold — a fold-level encoder leak found here was caught by a separate fold-scoping ablation and fixed (see note below). The two baseline rows are pre-fix and were not re-run.*

**On the leak that was found and fixed** — training-class aggregates (engineer historical mean, tier/channel/division/month means, city/state target encoding, segment delivery median) were originally fit on the full 2023-2025 cohort and split into train/validation without refitting, letting 2025 reach the validation rows. This inflated the earlier validation metrics and the apparent val→holdout gap. The fix scopes every aggregate to data available before the rows it serves; the holdout numbers above are leak-free.

**Segment-level AUC characteristic** — the model discriminates best in the worst-performing segments. That's exactly the right property for a resource allocation tool: it's most accurate where the decisions matter most.

**NPS validation (post-hoc, NPS never used in training; 2025 responders scored out-of-sample by the 2023-24 selection model)**

- **5.2-point gap** in promoter rate between low-risk and high-risk predicted buckets
- **3.8-point gap** in detractor rate between low-risk and high-risk predicted buckets

| Predicted-risk bucket | Promoter rate | Detractor rate |
|---|---:|---:|
| Very low (0-20% predicted late) | **62.6%** | 25.0% |
| Very high (80-100% predicted late) | **57.4%** | **28.8%** |
| **Gap** | **5.2pp** | **3.8pp** |

Predicted lateness tracks customer sentiment in the right direction — promoter rate falls and detractor rate rises from low- to high-risk segments — confirming the prioritization reflects real customer experience and not just a statistical artifact. The signal cannot be tuned from inside the pipeline because NPS is never a feature. (An earlier build reported a wider gap; that version was inflated by the encoder leak since corrected. Full 5-bucket breakdown: `outputs/prioritization/nps_validation.csv`, 42,451 responders.)

---

## Slide 8 — Priority Matrix: Where to Invest First

**Cases at Risk and Late Rate — Market Tier x Channel (T = 5 days)**

| Market Tier \ Channel | DMS | DMS2 | ASC | PP | ASD | AE | SPO |
|---|---|---|---|---|---|---|---|
| **Top 10** | 64,084 (37.7%) | 17,281 (38.3%) | 14,196 (41.1%) | 38,975 (44.5%) | 22,879 (76.5%) | 1,543 (68.9%) | 585 (87.1%) |
| **Metro** | 101,239 (45.5%) | 5,486 (49.6%) | 20,888 (53.2%) | 53,026 (57.9%) | 30,026 (56.2%) | 5,917 (71.3%) | 1,469 (76.0%) |
| **Urban** | 108,444 (49.5%) | 11,847 (63.0%) | 30,870 (56.2%) | 73,414 (62.4%) | 36,650 (73.1%) | 14,523 (77.9%) | 2,465 (90.2%) |
| **Rural** | 23,615 (57.2%) | 2,631 (46.5%) | 36,138 (63.8%) | 55,669 (68.0%) | 43,275 (72.1%) | 31,082 (79.7%) | 5,777 (87.7%) |

Numbers in parentheses are the late rate at T=5 (i.e., share of repairs in that segment that exceeded 5 days). These are observed late rates, unaffected by the model change.

**Top 5 priority segments**

| Rank | Segment | Cases at risk | Late rate | Primary lever |
|---|---|---:|---:|---|
| 1 | Urban x DMS | 108,444 | 49.5% | Parts Logistics |
| 2 | Metro x DMS | 101,239 | 45.5% | Parts Logistics |
| 3 | Urban x Premier Partner | 73,414 | 62.4% | Engineer Deployment |
| 4 | Top 10 x DMS | 64,084 | 37.7% | Parts Logistics |
| 5 | Rural x Premier Partner | 55,669 | 68.0% | Engineer Deployment |

<!-- VERIFY: primary-lever labels come from lever_decomposition.csv. The lever mix
shifted 46/25 -> 43/29 on re-run (one segment moved engineer -> parts), so confirm
these five primary-lever labels against the current outputs/prioritization/lever_decomposition.csv. -->

> **The top 5 priority segments cover 402,852 delayed cases — 47% of all at-risk repairs across the 28 analyzed segments** (total at-risk = 855,494).

---

## Slide 9 — Threshold Sensitivity: Priority Shifts from T=3 to T=10

**Priority rank shift across service targets** (observed late-rate based — unchanged)

| Segment | T=3 | T=5 | T=7 | T=10 |
|---|:-:|:-:|:-:|:-:|
| Urban x DMS | 1 | 1 | 2 | 6 |
| Metro x DMS | 2 | 2 | 3 | 8 |
| Urban x Premier Partner | 4 | **3** | **1** | **1** |
| Top 10 x DMS | 3 | 4 | 9 | - |
| Rural x Premier Partner | 6 | 5 | 4 | 3 |
| Rural x ASD | 8 | 7 | 6 | 2 |
| Metro x Premier Partner | 5 | 6 | 5 | 4 |
| Urban x ASD | 10 | 9 | 7 | 5 |

**LightGBM performance across thresholds — 2026 holdout (final model)**

| T | Holdout AUC | F1 | On-time rate |
|---|---:|---:|---:|
| T=3 | 0.787 | 0.232 | 27.9% |
| T=5 | 0.807 | 0.729 | 46.8% |
| T=7 | 0.827 | 0.842 | 62.3% |
| T=10 | 0.858 | 0.905 | 75.4% |

*(The right column is the on-time rate — share completed within T days. The earlier deck labeled this "Late Rate," which was the inverse and incorrect. Per-threshold precision/recall are in `outputs/models/threshold_results.csv`.)*

**Operational interpretation**

- DMS segments dominate at T=5 because a 5-day target is strict for their normal workflow. Most DMS repairs finish within 7 days, so high T=5 late rates don't necessarily indicate structural failure.
- At T=7, **Urban x Premier Partner rises to rank #1** with 50% of repairs still exceeding 7 days — signaling a persistent engineer deployment problem.
- Use T=3 and T=5 for standard warranty planning; use T=7 and T=10 for structurally complex repairs and longer-horizon investment decisions.

---

## Slide 10 — Action Agenda: Four Levers, Four Priorities

**From diagnosis to action**

| As-is | To-be |
|---|---|
| Uniform resource deployment | Segment-based prioritization |
| Delay drivers are mixed | Lever-specific intervention |
| Operational response is reactive | Planned operational action |

**Primary Lever Mix**

| Lever | Share of segments |
|---|---:|
| Engineer Deployment | 43% |
| Parts Logistics | 29% |
| Channel Process | 18% |
| Repair Complexity | 11% |

**Action 1 — Engineer Dispatch Rebalancing**

- *Target segments:* Urban x PP, Rural x PP, Rural x ASD, Urban x ASD, Rural x AE
- Rebalance more Q1/Q2 engineers into high-delay segments

**Action 2 — Parts Logistics Tail Reduction**

- *Target segments:* Urban x DMS, Metro x DMS, Top 10 x DMS
- Reduce delivery tails with pre-positioning and overnight shipment

**Action 3 — Channel Process Restructuring**

- *Target segments:* Top 10 x Premier Partner, SPO, AE
- Fix late channels through process changes, not blanket resources

**Action 4 — Repair Complexity Containment**

- *Target segments:* segments dominated by sealed-system and reclaim cases (3 of 28 segments, 11%)
- Improve first-time fix rate for sealed-system repairs; reduce reclaim rate through diagnostic quality improvements

---

## Slide 11 — Strategic Recommendations

> **RTAT delay is concentrated, not uniform — so intervention must be targeted.**

**Recommendation 1 — Engineer Deployment First**

- Reallocate stronger engineers to high-delay segments
- Prioritize Urban / Rural Premier Partner and ASD
- The **biggest reducible lever** — *85K improvable annually under conservative benchmarks across the top 10 segments*

**Recommendation 2 — Parts Logistics Second**

- Reduce delivery tails in DMS segments
- Prioritize Urban, Metro, and Top 10 DMS
- *273K cases at risk in DMS segments — targeting the 15% with 5-day-plus delivery tails eliminates the majority of late cases*

**Recommendation 3 — Channel Process Third**

- Fix structurally late channels
- Focus on SPO, AE, and Top 10 x Premier Partner
- *SPO + AE together: ~54,000 chronic late cases. Model AUC for SPO ≈ 0.58 — uniformly late, so risk scoring is insufficient. Structural renegotiation required.*

**Recommendation 4 — Repair Complexity Monitoring**

- Track sealed-system and reclaim rates as leading indicators
- Focus on the 3 segments where complexity is the primary lever (11% of segments analyzed)
- *First-time fix improvements compound: each prevented reclaim removes future delay from the pipeline*

**Operational direction shift**

| From | To |
|---|---|
| Average-based management | Segment-prioritized action |
| Blanket resource deployment | Targeted resource allocation based on primary lever |

---

## Appendix — Service Channel Glossary

The repair network operates across seven service channels reflecting ownership and credentialing structure:

| Code | Meaning | Approx mean RTAT |
|---|---|---:|
| **DMS** | Direct — in-house technician network | 6.0 days |
| **DMS2** | Direct — secondary in-house network | 6.5 days |
| **ASC** | Authorized Service Center | 8.5 days |
| **Premier Partner (PP)** | High-volume authorized third-party partners | 10.5 days |
| **ASD** | Authorized Service Distributor — regional third-party companies | 13.5 days |
| **AE** | Authorized Engineer — individual credentialed technicians | 16.5 days |
| **SPO** | Service Partner Other — residual partner network | 20.8 days |

The channels are ordered from fastest to slowest by mean RTAT in training data. Each channel has materially different operational characteristics — parts access, dispatch speed, geographic coverage, volume tier — which the model captures via channel-level mean RTAT and late-rate aggregates plus channel x market and channel x engineer-quartile interaction terms.

---

*This presentation was developed as a capstone analytics engagement for the Machine Learning & AI at Scale program at Emory University. All client-identifying details have been generalized to a "Fortune 500 appliance manufacturer" while preserving the methodology, findings, and aggregate results.*
