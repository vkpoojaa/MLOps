# PHASE 14 HANDOFF — ML/Data Monitoring

## 1. Objective

Phase 14 implements the ML/data monitoring requirement from the official Project 4 brief using the supplied incoming batches:

- Batch 1
- Batch 2
- Batch 3

The supplied files are treated as read-only. The monitor measures:

1. **Data/feature drift** against the historical training reference.
2. **Model performance drift** using the supplied actual `Price_Lakhs` values.

This layer is deliberately separate from Phase 13 operational monitoring (request count, latency, errors and prediction distribution).

A drift/performance alert **does not replace the production model**. Any retrained candidate must still pass the existing Data Gate and Model Gate before registry promotion.

---

## 2. Official-brief alignment

The official brief requires Task 7 to:

- serve the production model as a containerised API;
- monitor operational behaviour;
- process incoming batches for data drift;
- use supplied prices to measure performance drift;
- define justified alert thresholds;
- identify the batches that trigger alerts;
- explain the response when an alert fires.

Phase 13 covers the operational portion. Phase 14 covers the ML/data monitoring portion.

---

## 3. Reference data

### Training/reference dataset

`data/raw/capstone_house_prices_train.csv`

- 4,000 historical training rows.
- Same nine configured predictive features used by the production model.
- `Price_Lakhs` is the target and is **not** used as a feature-drift input.
- The reference is the full supplied training dataset.

### Incoming dataset

`data/raw/capstone_incoming_batches.csv`

- 1,500 rows total.
- 500 rows per batch.
- Contains actual `Price_Lakhs` for performance monitoring.
- `House_ID` and `Batch` are metadata, not predictive features.

The monitor computes the SHA-256 hash before and after processing. The hashes match, providing evidence that the supplied incoming file was not modified.

---

## 4. Feature-drift methodology

### Statistic

**Population Stability Index (PSI)** is calculated separately for each predictive feature.

PSI is calculated from the empirical reference and batch distributions rather than from a visual comparison.

### Numeric features

For each numeric feature:

- 10 quantile bins are defined from the full training reference.
- A separate missingness bin is included.
- The same reference bins are applied to every incoming batch.

Numeric features:

- `Area_sqft`
- `Bedrooms`
- `Bathrooms`
- `Age_Years`
- `Distance_to_Center_km`
- `Neighborhood_Score`
- `Has_Parking`

### Categorical features

The training-reference categories are used as fixed bins, plus a missingness category.

Categorical features:

- `Property_Condition`
- `City_Zone`

### Per-feature thresholds

Thresholds were **not chosen from a visual inspection or an arbitrary PSI rule of thumb**.

For each feature:

1. Draw 1,000 independent pseudo-batches of 500 rows from the training reference.
2. Calculate PSI for every pseudo-batch.
3. Use the empirical 95th percentile as that feature's null PSI threshold.

This estimates how large PSI can become from ordinary sampling variation when the incoming batch actually comes from the same distribution as training.

The resulting per-feature thresholds are:

| Feature | 95th-percentile null PSI |
|---|---:|
| Area_sqft | 0.034434 |
| Bedrooms | 0.014830 |
| Bathrooms | 0.011649 |
| Age_Years | 0.036973 |
| Distance_to_Center_km | 0.036841 |
| Neighborhood_Score | 0.038274 |
| Has_Parking | 0.000000 |
| Property_Condition | 0.015704 |
| City_Zone | 0.017444 |

A feature is considered drifted only when its observed PSI is above its calibrated feature-specific threshold.

### Batch-level threshold

A batch triggers the **data-drift alert** when at least 2 of the 9 predictive features are drifted, i.e. a drifted-feature share of at least:

`2 / 9 = 0.222222`

This was also calibrated empirically: the 95th percentile of the null drift-share distribution from the 1,000 bootstrap pseudo-batches was 2/9.

This prevents one noisy feature from automatically creating a batch-level drift alert.

---

## 5. Performance-drift methodology

The existing production artifact is used:

`artifacts/models/selected_model.joblib`

No model is retrained during monitoring.

### Reference performance

The model is evaluated on the same fixed seed-42 validation split used by the project training/evaluation methodology.

Reference validation metrics:

- MAE: **5.793117 lakhs**
- RMSE: **7.504375 lakhs**
- R²: **0.883088**

### Incoming performance

For each 500-row batch, predictions are generated using the existing production model and compared with the actual supplied `Price_Lakhs`.

Reported metrics:

- MAE
- RMSE
- R²
- relative MAE degradation versus the reference validation MAE

### Performance threshold

The performance threshold was also evidence-calibrated rather than selected visually.

- Generate 2,000 bootstrap pseudo-batches of 500 rows from the fixed validation set.
- Compute the pseudo-batch MAE.
- Compare it with the full validation MAE.
- Use the empirical 95th percentile of relative MAE degradation as the alert threshold.

Result:

**6.0281% relative MAE degradation**

A batch triggers performance drift when its relative MAE degradation is greater than this threshold.

---

## 6. Actual results

### Batch summary

| Batch | Data drift | Performance drift | Drifted features | MAE | RMSE | R² | Relative MAE degradation |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 1 | **NO** | **NO** | 0 | 5.5694 | 7.2290 | 0.8898 | -3.86% |
| 2 | **YES** | **YES** | 5 | 6.8813 | 9.0887 | 0.8566 | +18.78% |
| 3 | **YES** | **YES** | 5 | 13.1237 | 16.0676 | 0.6583 | +126.54% |

### Drifted features

**Batch 1:** none.

**Batch 2:**

- `Area_sqft`
- `Bedrooms`
- `Bathrooms`
- `Property_Condition`
- `City_Zone`

**Batch 3:**

- `Area_sqft`
- `Bedrooms`
- `Bathrooms`
- `Property_Condition`
- `City_Zone`

The exact PSI values and thresholds are stored in:

`reports/phase14_evidence/feature_drift.csv`

---

## 7. Business interpretation

### Batch 1 — stable

No feature-level drift exceeded its evidence-calibrated threshold, and performance improved slightly relative to the fixed validation reference.

**Interpretation:** the production model remains consistent with the training population for this batch. No retraining action is indicated from monitoring alone.

### Batch 2 — material change

Five features trigger drift. The most important changes are in property size/room characteristics and the categorical mix of property condition and city zone.

The production model's MAE increases from 5.7931 to **6.8813 lakhs**, an **18.78% degradation**, well above the 6.03% calibrated threshold.

The mean prediction bias is **-4.36 lakhs**, meaning the production model is systematically underpredicting this batch.

**Interpretation:** this is a meaningful population/business shift rather than a visual-only distribution difference. Investigation and retraining should be considered, especially if the pattern persists.

### Batch 3 — severe degradation

The same five features trigger drift, but the distribution shift is much stronger for property size and room counts and there is a substantial city-zone mix change.

MAE rises to **13.1237 lakhs**, a **126.54% degradation** versus the reference validation MAE.

The mean prediction bias is **-12.61 lakhs**.

**Interpretation:** this is a severe monitoring signal. The current model is materially underpredicting the incoming market segment. Retraining should be strongly considered after confirming that the batch represents a genuine business population change rather than a data-quality problem.

---

## 8. What should happen when an alert fires?

The response is intentionally conservative:

1. **Do not replace production automatically.**
2. Check the Data Gate and validate that the incoming batch is structurally and semantically acceptable.
3. Investigate the business reason for the shift.
4. If the shift is genuine and sustained, consider retraining.
5. Train a candidate using the established training pipeline.
6. Evaluate the candidate against the existing production model.
7. Require the existing Model Quality Gate to pass.
8. Register/promote only through the controlled registry pipeline.
9. Keep the previous production version available for rollback.

A monitoring alert therefore creates an **investigation/retraining signal**, not an uncontrolled model deployment.

---

## 9. Outputs for report, presentation and demo

Generated evidence:

- `reports/phase14_evidence/drift_results.json` — complete machine-readable results and methodology.
- `reports/phase14_evidence/batch_summary.csv` — one row per incoming batch.
- `reports/phase14_evidence/feature_drift.csv` — per-feature PSI and threshold evidence.
- `reports/phase14_evidence/phase14_monitoring_report.md` — presentation/report-ready summary.
- `reports/phase14_evidence/feature_drift_share.png` — batch drift-share chart.
- `reports/phase14_evidence/performance_drift.png` — relative MAE degradation chart.
- `reports/phase14_evidence/psi_heatmap.png` — PSI heatmap by feature and batch.

These values come from the supplied files and the existing production artifact; no dashboard/report values were manually fabricated.

---

## 10. Reproduction commands

From the repository root:

```bash
PYTHONPATH=src python -m mlops_pipeline validate-config
```

Run the complete incoming-batch monitor:

```bash
PYTHONPATH=src python -m mlops_pipeline monitor-batches
```

The default output directory is:

```text
reports/phase14_evidence/
```

Run Phase 14 tests:

```bash
PYTHONPATH=src pytest -q tests/unit/test_phase14_drift.py
```

The monitor reads the incoming CSV only; it does not write back to the supplied dataset.

---

## 11. Implementation

Primary implementation:

`src/mlops_pipeline/monitoring/drift.py`

CLI integration:

`python -m mlops_pipeline monitor-batches`

Configuration:

`config/config.yaml` under `monitoring:`

Tests:

`tests/unit/test_phase14_drift.py`

---

## 12. Verification performed

Phase 14 verification includes:

- configuration validation — PASS;
- Python compilation — PASS;
- supplied incoming-file hash unchanged before/after monitoring — PASS;
- all three supplied batches processed — PASS;
- actual target-based prediction metrics calculated — PASS;
- reproducible threshold/results test — PASS;
- Phase 14 tests — **3 passed**;
- unit suite — **237 passed**;
- integration suite — **32 passed**;
- combined unit + integration coverage — **269 passed**;
- Python compile check — PASS.

No live MLflow promotion or production-model replacement is performed by Phase 14.

---

## 13. Phase 15 instructions

Recommended Phase 15 objective: **final end-to-end integration and demonstration hardening**.

Implement and verify:

1. Connect the Phase 14 monitoring output to the overall Prefect lifecycle without bypassing the Data Gate or Model Gate.
2. Define the operational response path for an alert: log/record alert → investigate → optional retraining candidate → Model Gate → controlled registry promotion.
3. Demonstrate the safe behaviour for an alerting batch: the existing production model remains in service unless a separately trained candidate passes the Model Gate.
4. Connect the Phase 13 operational dashboard and Phase 14 drift/performance report into one presentation-ready monitoring story.
5. Prepare a 5–10 minute final demonstration covering:
   - data ingestion;
   - validation gate;
   - training/evaluation;
   - MLflow tracking/registry;
   - deployment/API;
   - Prometheus/Grafana operational monitoring;
   - Phase 14 batch drift/performance monitoring;
   - one failure/alert scenario;
   - safe model retention or rollback.
6. Capture final evidence/screenshots/logs and update the README with the complete runbook.
7. Do not auto-promote a retrained model solely because drift is detected. The existing Model Gate and controlled registry promotion remain mandatory.

---

**Phase 14 status: COMPLETE**

## Final audit remediation addendum

Phase 14 calibration controls are now configuration-driven. `monitoring.data_drift.bootstrap_runs`, `bootstrap_batch_size`, `monitoring.performance_drift.bootstrap_runs`, and `bootstrap_batch_size` are defined in `config/config.yaml` and validated by the typed schema. The monitoring implementation no longer contains fixed bootstrap-run constants. Operational p95 latency and error-rate alert ceilings are also configured in YAML.
