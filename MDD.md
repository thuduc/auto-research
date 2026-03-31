# Model Development Document

## Table of Contents
[1. Overview](#1-overview)  
[1.1 Executive Summary](#11-executive-summary)  
[1.2 Overview, Purpose, and Scope](#12-overview-purpose-and-scope)  
[1.3 Scope of Model Changes](#13-scope-of-model-changes)  
[1.4 Model Outputs](#14-model-outputs)  
[2. Modeling Data](#2-modeling-data)  
[2.1 Development Data Sources](#21-development-data-sources)  
[2.2 Data Cleaning, Filtering, and Transformation](#22-data-cleaning-filtering-and-transformation)  
[2.3 Development Data Quality Analysis](#23-development-data-quality-analysis)  
[2.4 Data Sampling](#24-data-sampling)  
[2.5 Data Limitations](#25-data-limitations)  
[3. Modeling Approach](#3-modeling-approach)  
[3.1 Model Methodology](#31-model-methodology)  
[3.2 Model Interactions & Dependencies](#32-model-interactions--dependencies)  
[3.3 Model Assumptions](#33-model-assumptions)  
[3.4 Limitations](#34-limitations)  
[4. Model Estimation & Specification](#4-model-estimation--specification)  
[4.1 Estimation Approach, Algorithm Selection, and Configuration](#41-estimation-approach-algorithm-selection-and-configuration)  
[4.2 Model Segmentation](#42-model-segmentation)  
[4.3 Variable/Feature Selection Process](#43-variablefeature-selection-process)  
[4.4 Model Selection](#44-model-selection)  
[4.5 Final Model Specification](#45-final-model-specification)  
[4.5.1 Ensemble Functional Form](#451-ensemble-functional-form)  
[4.5.2 Recursive Forecast Logic](#452-recursive-forecast-logic)  
[4.5.3 Input Variables Required for Production](#453-input-variables-required-for-production)  
[4.5.4 Component Model Configuration](#454-component-model-configuration)  
[4.5.5 Feature Importance and Relative Influence](#455-feature-importance-and-relative-influence)  
[4.5.6 Error Terms and Prediction Intervals](#456-error-terms-and-prediction-intervals)  
[4.5.7 Production Implementation Requirements](#457-production-implementation-requirements)  
[4.5.8 Artifact and Provenance References](#458-artifact-and-provenance-references)  
[4.5.9 Step-by-Step Production Algorithm](#459-step-by-step-production-algorithm)  
[4.5.10 Forecast-Cycle Pseudocode](#4510-forecast-cycle-pseudocode)  
[4.5.11 Production Input / Output Contract](#4511-production-input--output-contract)  
[4.6 In-Model Overlays](#46-in-model-overlays)  
[5. Model Testing](#5-model-testing)  
[5.1 Model Development Test Plan & Approach](#51-model-development-test-plan--approach)  
[5.2 Backtesting](#52-backtesting)  
[5.3 Segmentation Testing](#53-segmentation-testing)  
[5.4 Sensitivity and Feature Importance Testing](#54-sensitivity-and-feature-importance-testing)  
[5.5 Scenario Analysis & Stress Testing](#55-scenario-analysis--stress-testing)  
[5.6 Benchmarking](#56-benchmarking)  
[5.7 Assumptions Analysis](#57-assumptions-analysis)  
[5.8 In-Model Overlay Testing](#58-in-model-overlay-testing)  
[6. Model Governance](#6-model-governance)  
[6.1 Model Risk Considerations](#61-model-risk-considerations)  
[6.2 Monitoring and Ongoing Performance Review](#62-monitoring-and-ongoing-performance-review)  
[6.3 Implementation and Control Considerations](#63-implementation-and-control-considerations)  
[7. Appendix](#7-appendix)  
[7.1 Glossary](#71-glossary)  
[7.2 Model Code & Dataset Location](#72-model-code-dataset-location)  
[7.3 References](#73-references)  
[7.4 Change Log](#74-change-log)  
[7.5 Data Dictionary](#75-data-dictionary)

## 1. Overview

### 1.1 Executive Summary
The current champion candidate is a finalized fixed-weight ensemble of `Gradient Boosting` and `XGBoost`. It was selected after a governed search across econometric, machine learning, deep learning, hybrid, and ensemble families and achieved the best observed development score in the repository, `GOF_composite = 0.642211`. Relative to the strongest single-model challenger, the final `60/40` ensemble improved the model frontier while preserving simple and reviewable production logic.

From a model-risk perspective, the champion is attractive because it avoids opaque adaptive reweighting and remains traceable to two persisted governed component specifications. The most important current residual concerns are incomplete business-governance metadata and the fact that the final holdout period shows systematic underprediction in a rising end-of-sample environment.

### 1.2 Overview, Purpose, and Scope
This document describes `National HPI Champion Ensemble`, the current finalized national HPI champion candidate. The model was selected through an autonomous but governed research process that evaluated econometric, machine learning, deep learning, hybrid, and ensemble families under a common scoring framework. The finalized champion is a `Model Ensemble` using a fixed weighted-average combination of `Gradient Boosting` and `XGBoost`, and its intended use is `Quarterly national HPI forecasting for governed research and production candidate evaluation.`.

The finalized champion artifact is `20260331_ensemble_gb_xgb_finalize / 20260331_ensemble_gb_xgb_finalize_001`. Its development score is `GOF_composite = 0.642211`, compared with the strongest single-model challenger, `Gradient Boosting`, at `GOF_composite = 0.637548`. This means the final ensemble improved the frontier while preserving relatively simple and reviewable combination logic.

### 1.3 Scope of Model Changes
The scope of model change documented here includes the full breadth-first champion search, subsequent full-20 variable-selection reruns, system-model reruns, hybrid experimentation, and final ensemble refinement. Relative to the original single-model frontier, the final controlled change was replacement of the strongest individual challenger with a two-component fixed-weight ensemble. The document therefore covers both the final approved champion and the decision path that justified that promotion.

### 1.4 Model Outputs
The model produces a quarterly national HPI forecast path, associated interval forecasts, validation and holdout diagnostics, and the model-development evidence needed to support governance review. In operational terms the output package includes a finalized ensemble specification, component provenance, benchmark comparisons, feature-importance summaries, and development documentation artifacts that can be used to populate the official model development document.

## 2. Modeling Data

### 2.1 Development Data Sources
Development data comes from FHFA via FRED for the HPI target and from FRED for the macroeconomic, housing, financing, and price indicators used during champion search. The canonical data definitions are governed by `config/variables.yaml`, while the realized transformed dataset is captured in `data/processed/search_panel.parquet`, `data/processed/holdout_panel.parquet`, and `data/processed/dataset_manifest.json`. The resulting development dataset supports a long historical search sample and a separate final holdout window reserved for post-selection evaluation.

### 2.2 Data Cleaning, Filtering, and Transformation
The data pipeline is intentionally frozen in `prepare.py` and is therefore separated from mutable model-research logic. Each series is aligned to quarterly frequency, transformed according to its declared rule, and then integrated into a common search panel. The transformation policy includes a mix of levels, log differences, and log-level transformations depending on the economics of the underlying series. Structural dummies such as the GFC and COVID indicators are prepared centrally as deterministic helper variables. The resulting design ensures that all model classes, from econometric families through deep learning and ensembles, are evaluated against the same transformed substrate.

### 2.3 Development Data Quality Analysis
The search panel contains `194` rows and `23` columns. Coverage review indicates that the included development variables have full non-null support over the realized search window used by the transformed dataset. Coverage diagnostics are recorded in `data/processed/data_quality_report.json`, while collinearity diagnostics are recorded in `data/processed/collinearity_report.json`. Representative coverage rows are shown below.

| column | non_null_rows | missing_rate | first_valid_date | last_valid_date |
| --- | --- | --- | --- | --- |
| hpi | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| real_gdp | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| per_capita_income | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| population | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| unemployment_rate | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| consumer_confidence | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| personal_consumption | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| housing_starts | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| building_permits | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |
| new_home_sales | 194 | 0.0000 | 1976-09-30 | 2024-12-31 |

Representative high-correlation pairs are shown below. These do not automatically invalidate variables, but they provide important context for model-family behavior, especially in econometric families that are more sensitive to multicollinearity.

| left | right | correlation |
| --- | --- | --- |
| real_gdp | personal_consumption | 0.8603 |
| mortgage_rate | treasury_10y | 0.9900 |
| mortgage_rate | fed_funds | 0.9376 |
| treasury_10y | fed_funds | 0.9278 |
| cpi_all_items | cpi_less_shelter | 0.9543 |

### 2.4 Data Sampling
The transformed search sample spans `1976-09-30` through `2024-12-31`, with the primary search window beginning at `1995-03-31`. The final holdout sample spans `2025-03-31` through `2025-12-31`. Development backtesting uses annual origins from `2005-12-31` through `2014-12-31`, which provides repeated near-horizon and far-horizon evaluation windows while preserving the holdout for post-selection review.

### 2.5 Data Limitations
Important data limitations remain even after transformation and quality review. First, not every configured variable survives the primary search-window history requirement. Second, long historical samples can mix multiple housing and monetary regimes, which can weaken the stability of simple cross-period relationships. Third, transformed data is not the same as vintage-real-time production information, so measured development performance should not be interpreted as a perfect proxy for live historical decision-time performance. The excluded-variable log for the current dataset is: 
- `existing_home_sales` was excluded because `insufficient_history_for_primary_search_window`; earliest valid date was `2025-03-31`
- `sp500` was excluded because `insufficient_history_for_primary_search_window`; earliest valid date was `2016-06-30`

## 3. Modeling Approach

### 3.1 Model Methodology
The finalized champion is a simple, fixed-weight model ensemble composed of two tabular tree-based forecasting components: `Gradient Boosting` and `XGBoost`. Both component models forecast quarterly national HPI from lagged HPI and a screened exogenous feature set. The ensemble does not use adaptive online learning, recursive reweighting, or black-box stacking logic at production time. Instead, it uses a fixed weighted average, selected during refinement because it outperformed the standalone components and other ensemble constructions while remaining easy to explain and reproduce.

### 3.2 Model Interactions & Dependencies
The champion depends on two persisted component specifications and their governed artifacts. These dependencies are intentional and explicit rather than implicit. The ensemble layer does not replace component governance; it adds a documented combination rule over already persisted and reproducible base forecasts. The finalized dependencies are:
- `Gradient Boosting` from `20260330_gradient_boosting_full20_300 / 20260330_gradient_boosting_full20_300_091` weight=0.6: Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4] with consumer_confidence
- `XGBoost` from `20260330_xgboost_full20_300 / 20260330_xgboost_full20_300_068` weight=0.4: XGBoost n_estimators=500 learning_rate=0.05 max_depth=3 subsample=0.8 colsample_bytree=0.8 lags=[1, 2, 4] with consumer_confidence
The ensemble method is `weighted_average` with frozen weights of `0.60` on `Gradient Boosting` and `0.40` on `XGBoost`.

### 3.3 Model Assumptions
- Historical quarterly HPI and macro relationships are informative for future forecasting over the governed use horizon.
- The finalized ensemble uses fixed weights and does not adapt online after model approval.
- Forecast quality depends on the continued informativeness of lagged HPI and consumer_confidence, which dominate the selected component specifications.
- The validation framework based on annual rolling origins from 2005-12-31 through 2014-12-31 remains representative enough for model ranking.
- The national aggregate model is intended for national HPI forecasting and does not imply regional segmentation validity.
These assumptions are not merely academic. They explain why the model is expected to remain useful after selection and identify the conditions under which performance deterioration would be plausible.

### 3.4 Limitations
- No in-model overlays are present in the finalized champion.
- Model relationships may degrade under structural breaks not well represented in the search sample.
An additional practical limitation is that the final model remains a national aggregate forecaster. It should not be interpreted as validated for regional segmentation, loan-level use, or any other out-of-scope application without separate development and validation. The current stress framework is implemented, but it remains mechanically derived rather than a curated supervisory scenario set.

## 4. Model Estimation & Specification

### 4.1 Estimation Approach, Algorithm Selection, and Configuration
Model estimation followed a governed multi-stage search process. First, broad candidate experiments were run across econometric, machine learning, deep learning, hybrid, and ensemble families. Second, the strongest exogenous families were rerun under a more rigorous full-20 variable-selection framework. Third, multivariate system families were rerun under an analogous system-selection framework. Fourth, promising hybrids and ensembles were evaluated using persisted validation outputs rather than ad hoc manual combinations. Finally, the strongest `Gradient Boosting` / `XGBoost` blend was refined locally around the best discovered weighting region.

This process is important because the final champion was not selected from a narrow local search. It won after surviving a wide family sweep, variable-selection reruns, and dedicated ensemble refinement.

### 4.2 Model Segmentation
No model segmentation is used. The entire development process targets a single national HPI series. This simplifies the final specification and avoids unsupported claims about segment-specific calibration.

### 4.3 Variable/Feature Selection Process
Variable selection evolved materially during the project. Early candidate generation relied more on curated shortlists. Later runs moved to a full-20 screening framework built from the configured exogenous universe. That framework materially improved the strongest ML and some deep-learning families. For the final champion, both winning component models converged on a common high-performing structure: lagged HPI terms `[1, 2, 4]` plus the single screened exogenous variable `consumer_confidence`. The fact that both final components independently retained the same exogenous input gives additional qualitative support to the final simplified feature set.

### 4.4 Model Selection
Model selection was governed by the composite GOF framework, which combines in-sample fit, near-horizon validation, far-horizon validation, and diagnostics. The final ensemble was selected because it achieved the best observed composite score while preserving relatively transparent logic compared with more elaborate stacking approaches. The model-family frontier is summarized below.

| family | best_model_class | gof_composite | description |
| --- | --- | --- | --- |
| econometric_single_equation | ARIMAX | 0.2116 | ARIMAX(1, 1, 0) trend=t with treasury_10y, housing_inventory, new_home_sales |
| econometric_system | VECM | 0.2456 | VECM(k_ar_diff=1) deterministic=co with building_permits |
| tabular_ml | Gradient Boosting | 0.6375 | Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4] with consumer_confidence |
| deep_learning | N-BEATS | 0.5856 | N-BEATS lookback=4 stack_width=96 n_blocks=2 n_layers=2 epochs=75 learning_rate=0.01 with housing_inventory |
| hybrid_ensemble | Model Ensemble | 0.6422 | static 60/40 ensemble of Gradient Boosting, XGBoost |

The overall leaderboard frontier is shown below.

| rank | experiment_id | model_class | gof_composite | description |
| --- | --- | --- | --- | --- |
| 1 | 20260331_ensemble_gb_xgb_refine | Model Ensemble | 0.6422114864199716 | static 60/40 ensemble of Gradient Boosting, XGBoost |
| 2 | 20260331_ensemble_gb_xgb_finalize | Model Ensemble | 0.6422114864199716 | Finalized static 60/40 ensemble of Gradient Boosting, XGBoost |
| 3 | 20260331_ensemble_gb_xgb_full20 | Model Ensemble | 0.6412762330528322 | inverse_rmse ensemble of Gradient Boosting, XGBoost |
| 4 | 20260330_gradient_boosting_full20_300 | Gradient Boosting | 0.6375482073353181 | Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4] with consumer_confidence |
| 5 | 20260331_gradient_boosting_refine_60 | Gradient Boosting | 0.6375482073353181 | Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4] with consumer_confidence |
| 6 | 20260331_ensemble_top4_full20 | Model Ensemble | 0.6318170949122243 | inverse_rmse ensemble of Gradient Boosting, XGBoost, N-BEATS, Transformer |
| 7 | 20260330_xgboost_full20_300 | XGBoost | 0.6252645788374875 | XGBoost n_estimators=500 learning_rate=0.05 max_depth=3 subsample=0.8 colsample_bytree=0.8 lags=[1, 2, 4] with consumer_confidence |
| 8 | 20260331_xgboost_refine_60 | XGBoost | 0.6239836628960995 | XGBoost n_estimators=300 learning_rate=0.05 max_depth=4 subsample=0.7 colsample_bytree=0.7 lags=[1, 2, 4] with consumer_confidence |
| 9 | 20260331_nbeats_refine_60 | N-BEATS | 0.5855872642341936 | N-BEATS lookback=4 stack_width=96 n_blocks=2 n_layers=2 epochs=75 learning_rate=0.01 with housing_inventory |
| 10 | 20260329_xgboost_300 | XGBoost | 0.565112281218815 | XGBoost n_estimators=500 learning_rate=0.1 max_depth=2 subsample=0.8 colsample_bytree=0.8 lags=[1, 4, 8] with no exogenous variables |
| 11 | 20260329_xgboost_finalize | XGBoost | 0.565112281218815 | XGBoost n_estimators=500 learning_rate=0.1 max_depth=2 subsample=0.8 colsample_bytree=0.8 lags=[1, 4, 8] with no exogenous variables |
| 12 | 20260329_gradient_boosting_300 | Gradient Boosting | 0.5604065322595904 | Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4, 8] with no exogenous variables |

### 4.5 Final Model Specification
#### 4.5.1 Ensemble Functional Form
Let `GB_t` denote the quarter-`t` point forecast from the finalized `Gradient Boosting` component and `XGB_t` denote the quarter-`t` point forecast from the finalized `XGBoost` component. The production point forecast is:

$$
\hat{HPI}_t = 0.60 \cdot GB_t + 0.40 \cdot XGB_t
$$

The same fixed weights are applied to the lower and upper interval endpoints:

$$
L^{90}_t = 0.60 \cdot L^{90,GB}_t + 0.40 \cdot L^{90,XGB}_t
U^{90}_t = 0.60 \cdot U^{90,GB}_t + 0.40 \cdot U^{90,XGB}_t
L^{50}_t = 0.60 \cdot L^{50,GB}_t + 0.40 \cdot L^{50,XGB}_t
U^{50}_t = 0.60 \cdot U^{50,GB}_t + 0.40 \cdot U^{50,XGB}_t
$$

#### 4.5.2 Recursive Forecast Logic
Each component is forecast recursively. For quarter `t`, the feature vector is built from lagged HPI values and the quarter-`t` exogenous input. If a lag refers to a forecasted quarter rather than an observed quarter, the previously generated model forecast is used. In implementation terms, production forecasting must append each newly predicted `hpi` value back into the history buffer before generating the next quarter.

$$
x_t = \left[hpi_{t-1},\; hpi_{t-2},\; hpi_{t-4},\; consumer\_confidence_t\right]
GB_t = f_{GB}(x_t)
XGB_t = f_{XGB}(x_t)
\hat{HPI}_t = 0.60 \cdot GB_t + 0.40 \cdot XGB_t
$$

#### 4.5.3 Input Variables Required for Production
| input_variable | description | source |
| --- | --- | --- |
| hpi_lag_1 | Most recent observed or recursively predicted quarterly HPI level | target history |
| hpi_lag_2 | Second most recent observed or recursively predicted quarterly HPI level | target history |
| hpi_lag_4 | Observed or recursively predicted HPI level four quarters prior | target history |
| consumer_confidence | Quarterly consumer sentiment / confidence input after repo-defined transformation | prepared exogenous path |

#### 4.5.4 Component Model Configuration
| component | target_lags | exogenous | configuration |
| --- | --- | --- | --- |
| Gradient Boosting | [1, 2, 4] | consumer_confidence | n_estimators=500, learning_rate=0.05, max_depth=3, min_samples_leaf=1, subsample=1.0 |
| XGBoost | [1, 2, 4] | consumer_confidence | n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1.0 |

#### 4.5.5 Feature Importance and Relative Influence
Feature importances are not coefficients and should not be interpreted as linear marginal effects, but they are useful implementation evidence because they confirm the effective feature set and the dominant role of lagged HPI in both final components.

| component | feature | importance |
| --- | --- | --- |
| Gradient Boosting | hpi_lag_1 | 0.531777 |
| Gradient Boosting | hpi_lag_2 | 0.269166 |
| Gradient Boosting | hpi_lag_4 | 0.197996 |
| Gradient Boosting | consumer_confidence | 0.001061 |
| XGBoost | hpi_lag_1 | 0.818230 |
| XGBoost | hpi_lag_2 | 0.173409 |
| XGBoost | hpi_lag_4 | 0.007379 |
| XGBoost | consumer_confidence | 0.000983 |

#### 4.5.6 Error Terms and Prediction Intervals
Each tabular component produces prediction intervals using a residual-standard-deviation approximation with horizon scaling proportional to `sqrt(h)`. For a forecast horizon `h` and component residual standard deviation `sigma`, the component interval width is:

$$
w(\alpha, h) = z_{1-\alpha/2} \cdot \sigma \cdot \sqrt{h}
$$

where `alpha = 0.10` for 90% intervals and `alpha = 0.50` for 50% intervals. The ensemble combines component interval endpoints using the same `0.60 / 0.40` weights. This is a pragmatic interval-construction rule rather than a full probabilistic dependence model.

#### 4.5.7 Production Implementation Requirements
- Use the frozen component artifacts referenced in the finalized ensemble specification.
- Construct quarterly features exactly as `[hpi_lag_1, hpi_lag_2, hpi_lag_4, consumer_confidence]`.
- Apply the recursive update logic quarter by quarter.
- Apply fixed weights `0.60` and `0.40` to the two component forecasts and interval endpoints.
- Use the prepared exogenous path logic or approved scenario-path logic from the governed codebase.
- Treat any change to weights, lags, feature transforms, or component hyperparameters as a model change.

#### 4.5.8 Artifact and Provenance References
- Finalized ensemble specification: `/Users/duc/projects/claude/github/fanniemae-cc-poc/auto-research-hpf/experiments/20260331_ensemble_gb_xgb_finalize/runs/20260331_ensemble_gb_xgb_finalize_001/spec.json`
- Finalized ensemble metrics: `/Users/duc/projects/claude/github/fanniemae-cc-poc/auto-research-hpf/experiments/20260331_ensemble_gb_xgb_finalize/runs/20260331_ensemble_gb_xgb_finalize_001/metrics.json`
- Gradient Boosting component spec: `experiments/20260330_gradient_boosting_full20_300/runs/20260330_gradient_boosting_full20_300_091/spec.json`
- XGBoost component spec: `experiments/20260330_xgboost_full20_300/runs/20260330_xgboost_full20_300_068/spec.json`
- Scenario forecast output: `output/forecasts/champion_forecast_scenarios.csv`

#### 4.5.9 Step-by-Step Production Algorithm
- Load the frozen finalized ensemble specification and the two referenced component model artifacts.
- Load the latest approved quarterly HPI history and the required exogenous input path for each forecast quarter.
- For each forecast quarter `t`, construct the feature vector `[hpi_lag_1, hpi_lag_2, hpi_lag_4, consumer_confidence_t]` using observed HPI history where available and previously predicted HPI values once the forecast horizon moves beyond observed history.
- Score the feature vector with the finalized Gradient Boosting model to obtain `GB_t`.
- Score the same feature vector with the finalized XGBoost model to obtain `XGB_t`.
- Compute the ensemble point forecast `HPI_hat_t = 0.60 * GB_t + 0.40 * XGB_t`.
- Compute each component interval endpoint using its residual-standard-deviation rule and then combine the interval endpoints with the same `0.60 / 0.40` weights.
- Append `HPI_hat_t` to the internal history buffer so the next quarter can reference it in the lag structure.
- Repeat until all requested forecast quarters are produced.
- Persist point forecasts, interval forecasts, scenario identifier if applicable, and artifact provenance for auditability.

#### 4.5.10 Forecast-Cycle Pseudocode
```text
inputs:
  history_hpi[1..T]
  future_consumer_confidence[1..H]
  GB_model, XGB_model
  weights = {gb: 0.60, xgb: 0.40}

for h in 1..H:
    x_h = [
        hpi_lag_1 = last(history_hpi, 1),
        hpi_lag_2 = last(history_hpi, 2),
        hpi_lag_4 = last(history_hpi, 4),
        consumer_confidence = future_consumer_confidence[h]
    ]

    gb_pred  = GB_model.predict(x_h)
    xgb_pred = XGB_model.predict(x_h)

    final_pred = 0.60 * gb_pred + 0.40 * xgb_pred

    gb_interval  = interval(gb_pred, gb_sigma, h)
    xgb_interval = interval(xgb_pred, xgb_sigma, h)

    final_lower_90 = 0.60 * gb_interval.lower_90 + 0.40 * xgb_interval.lower_90
    final_upper_90 = 0.60 * gb_interval.upper_90 + 0.40 * xgb_interval.upper_90
    final_lower_50 = 0.60 * gb_interval.lower_50 + 0.40 * xgb_interval.lower_50
    final_upper_50 = 0.60 * gb_interval.upper_50 + 0.40 * xgb_interval.upper_50

    append(final_pred to history_hpi)
    write forecast row for horizon h
```

#### 4.5.11 Production Input / Output Contract
**Required inputs**
- Quarterly timestamp sequence for the forecast horizon.
- Observed historical national HPI values through the forecast origin date.
- Quarterly `consumer_confidence` values for each forecast quarter under baseline or approved stress scenario assumptions.
- Frozen component model artifacts for the finalized Gradient Boosting and XGBoost models.
- Frozen ensemble specification with component references and weights.

**Expected outputs**
- One forecast row per horizon quarter.
- Columns: `date`, `hpi_forecast`, `hpi_lower_90`, `hpi_upper_90`, `hpi_lower_50`, `hpi_upper_50`.
- For scenario runs, include a `scenario` identifier.
- For auditability, retain the component experiment IDs, trial IDs, and ensemble weights used in the run metadata.

**Operational constraints**
- Forecast generation is quarterly and recursive.
- Missing required inputs should fail the run rather than silently impute unsupported values.
- Any substitution of feature transforms, lag structure, component models, or weights requires formal model-change handling.

### 4.6 In-Model Overlays
No in-model overlays were used in the finalized champion. This simplifies governance because the final forecast is produced directly by the underlying component models and the documented ensemble rule, without judgmental adjustments layered on top.

## 5. Model Testing

### 5.1 Model Development Test Plan & Approach
Model testing was designed to be common across heterogeneous model families. Development ranking used repeated rolling-origin validation on the search sample, with explicit separation between near-horizon and far-horizon performance. Diagnostics and plausibility checks supplemented error-based metrics. The final holdout was reserved for post-selection review and was not used to choose the champion during search.

### 5.2 Backtesting
For the finalized champion, development-stage validation produced `gof_insample = 0.964403`, `gof_validation_near = 0.415620`, `gof_validation_far = 0.101001`, and `gof_diagnostic = 0.958333`. The overall composite score was `GOF_composite = 0.642211`. These results indicate a model that is very strong in-sample, strongest in the nearer validation horizons, and still positive but weaker in the far-horizon validation range.

Post-selection holdout review produced `rmse = 17.399081` and `mae = 16.101210` over `4` quarters. The holdout path is shown below.

| date | actual | forecast | error | inside_90 | inside_50 |
| --- | --- | --- | --- | --- | --- |
| 2025-03-31 | 691.36 | 685.79 | 5.57 | False | False |
| 2025-06-30 | 701.82 | 685.79 | 16.03 | False | False |
| 2025-09-30 | 705.32 | 685.79 | 19.53 | False | False |
| 2025-12-31 | 709.05 | 685.79 | 23.26 | False | False |

### 5.3 Segmentation Testing
Not applicable. Because the model is developed for a single national aggregate target, there are no segment-specific performance claims to validate in the current document.

### 5.4 Sensitivity and Feature Importance Testing
Sensitivity evidence is currently strongest at the component level. For both component tree models, lagged HPI variables dominate raw feature-importance mass, while `consumer_confidence` appears as a retained but low-magnitude exogenous signal. This should not be over-interpreted as meaning `consumer_confidence` is unimportant in development terms; rather, it means that once the model already has lagged HPI, the incremental explanatory contribution assigned by the fitted tree structures is small relative to autoregressive structure. The feature-importance summary is shown below.

| component | feature | importance |
| --- | --- | --- |
| Gradient Boosting | hpi_lag_1 | 0.531777 |
| Gradient Boosting | hpi_lag_2 | 0.269166 |
| Gradient Boosting | hpi_lag_4 | 0.197996 |
| Gradient Boosting | consumer_confidence | 0.001061 |
| XGBoost | hpi_lag_1 | 0.818230 |
| XGBoost | hpi_lag_2 | 0.173409 |
| XGBoost | hpi_lag_4 | 0.007379 |
| XGBoost | consumer_confidence | 0.000983 |

### 5.5 Scenario Analysis & Stress Testing
Scenario generation is now available for the finalized ensemble champion. The current implementation produces `baseline`, `adverse`, and `severely_adverse` scenario paths by applying stressed exogenous trajectories to the finalized `Gradient Boosting` and `XGBoost` component models and then recombining those component forecasts into the final forecast `\hat{HPI}_t` using the frozen `0.60 / 0.40` ensemble weights. The resulting scenario summaries are shown below.

| scenario | implemented | start_forecast | q8_forecast | q20_forecast | q40_forecast | end_forecast |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | True | 685.786 | 685.786 | 685.346 | 685.346 | 685.346 |
| adverse | True | 685.652 | 685.401 | 685.401 | 685.346 | 685.346 |
| severely_adverse | True | 685.401 | 683.563 | 683.563 | 685.346 | 685.346 |

The current stress results behave directionally as expected: the adverse and severely adverse paths sit below the baseline path in the early and middle forecast horizons, while long-horizon values revert toward the same long-run baseline anchor as the stressed exogenous variables mean-revert. This is a mechanically derived stress framework rather than a macroeconomically curated supervisory scenario set, but it is materially stronger than the prior placeholder-only state.

### 5.6 Benchmarking
Benchmarking was performed continuously during champion search. The final ensemble was compared not only to weaker benchmarks, but also to the strongest single-model challengers and to other hybrid / ensemble variants. The current benchmark frontier is shown below.

| experiment_id | model_class | gof_composite | description |
| --- | --- | --- | --- |
| 20260331_ensemble_gb_xgb_refine | Model Ensemble | 0.6422114864199716 | static 60/40 ensemble of Gradient Boosting, XGBoost |
| 20260331_ensemble_gb_xgb_finalize | Model Ensemble | 0.6422114864199716 | Finalized static 60/40 ensemble of Gradient Boosting, XGBoost |
| 20260331_ensemble_gb_xgb_full20 | Model Ensemble | 0.6412762330528322 | inverse_rmse ensemble of Gradient Boosting, XGBoost |
| 20260330_gradient_boosting_full20_300 | Gradient Boosting | 0.6375482073353181 | Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4] with consumer_confidence |
| 20260331_gradient_boosting_refine_60 | Gradient Boosting | 0.6375482073353181 | Gradient Boosting n_estimators=500 learning_rate=0.05 max_depth=3 min_samples_leaf=1 lags=[1, 2, 4] with consumer_confidence |
| 20260331_ensemble_top4_full20 | Model Ensemble | 0.6318170949122243 | inverse_rmse ensemble of Gradient Boosting, XGBoost, N-BEATS, Transformer |
| 20260330_xgboost_full20_300 | XGBoost | 0.6252645788374875 | XGBoost n_estimators=500 learning_rate=0.05 max_depth=3 subsample=0.8 colsample_bytree=0.8 lags=[1, 2, 4] with consumer_confidence |
| 20260331_xgboost_refine_60 | XGBoost | 0.6239836628960995 | XGBoost n_estimators=300 learning_rate=0.05 max_depth=4 subsample=0.7 colsample_bytree=0.7 lags=[1, 2, 4] with consumer_confidence |
| 20260331_nbeats_refine_60 | N-BEATS | 0.5855872642341936 | N-BEATS lookback=4 stack_width=96 n_blocks=2 n_layers=2 epochs=75 learning_rate=0.01 with housing_inventory |
| 20260329_xgboost_300 | XGBoost | 0.565112281218815 | XGBoost n_estimators=500 learning_rate=0.1 max_depth=2 subsample=0.8 colsample_bytree=0.8 lags=[1, 4, 8] with no exogenous variables |

### 5.7 Assumptions Analysis
Assumptions analysis is documented in `output/diagnostics/assumptions_report.json`. In practice, the most important assumptions are that lagged HPI remains highly informative, that `consumer_confidence` continues to add directional value, and that the rolling-origin design used during development remains a reasonable proxy for out-of-sample deployment behavior. The holdout results already suggest that the final champion may underpredict in a rising end-of-sample environment, so this is an area to monitor carefully even though the model remains the best development-stage candidate.

### 5.8 In-Model Overlay Testing
Not applicable. No overlays were applied, so no overlay-specific testing was required. This removes one common source of governance ambiguity, because the final forecast can be traced directly to model components and documented combination logic.

## 6. Model Governance

The current governance metadata is stored in `config/model_governance.yaml`. The model is identified as `National HPI Champion Ensemble` with version `2026-03-31` and implementation status `candidate_finalized`. The intended use is `Quarterly national HPI forecasting for governed research and production candidate evaluation.`. The configured review cadence is `Quarterly`.

Model owner: `TBD`

Business owner: `TBD`

Approvers:
- TBD

Prohibited use:
- Regional or loan-level decisioning without separate validation
- Use outside the documented quarterly national HPI forecasting scope

### 6.1 Model Risk Considerations
The principal model-risk considerations at this stage are performance stability, end-of-sample bias, and governance completeness. Development evidence strongly supports the selected champion relative to all evaluated alternatives, but the holdout review indicates a consistent underprediction pattern across the four reserved holdout quarters. That does not negate champion status, but it does indicate that production monitoring should pay close attention to bias drift and sustained forecast underestimation in rising market conditions.

A second model-risk consideration is implementation scope. The model is validated for national aggregate HPI forecasting only and should not be repurposed for regional, loan-level, or decisioning applications without separate validation. A third consideration is that the stress framework is now implemented, but it remains a mechanically derived scenario engine rather than a curated supervisory scenario set.

### 6.2 Monitoring and Ongoing Performance Review
At minimum, ongoing monitoring should review realized forecast error, directional accuracy, interval coverage, and stability of the final ensemble relative to its component models. Current holdout metrics are `rmse = 17.399081`, `mae = 16.101210`, `coverage_90 = 0.0000`, and `coverage_50 = 0.0000`. Recommended monitoring triggers include sustained forecast bias, persistent interval undercoverage, material deterioration versus naive or prior champion benchmarks, and evidence that the component models diverge materially from one another in live forecasting.

Periodic review should also confirm that the governed variable universe, transformation rules, and component model artifacts remain unchanged except through approved model-change processes.

### 6.3 Implementation and Control Considerations
Implementation controls should ensure that production uses the frozen component model artifacts and the documented `60/40` ensemble rule only. Any change to component specifications, exogenous construction, combination weights, scenario logic, or data-preparation assumptions should be treated as model change rather than ordinary runtime operation.

The current document is operationally strong for technical review, but governance completion still requires organization-specific owner and approver values to replace the current placeholders.

## 7. Appendix

### 7.1 Glossary
- Experiment: an autonomous search loop for one model family or composition type
- Trial: one fully specified candidate evaluated within an experiment
- Champion: the current best governed model candidate after cross-family comparison
- GOF composite: weighted development score combining fit, validation, and diagnostics
- Holdout: final reserved sample used only after model selection
- Full-20 framework: variable-screening process over the broad configured exogenous universe

### 7.2 Model Code & Dataset Location
- Finalized champion run directory: `/Users/duc/projects/claude/github/fanniemae-cc-poc/auto-research-hpf/experiments/20260331_ensemble_gb_xgb_finalize/runs/20260331_ensemble_gb_xgb_finalize_001`
- Search panel: `data/processed/search_panel.parquet`
- Holdout panel: `data/processed/holdout_panel.parquet`
- Data quality report: `data/processed/data_quality_report.json`
- Collinearity report: `data/processed/collinearity_report.json`
- Champion report: `output/diagnostics/champion_report.json`
- Feature importance report: `output/diagnostics/feature_importance_report.json`
- Assumptions report: `output/diagnostics/assumptions_report.json`

### 7.3 References
- FHFA HPI via FRED: https://fred.stlouisfed.org/series/USSTHPI
- real_gdp: https://fred.stlouisfed.org/series/GDPC1
- per_capita_income: https://fred.stlouisfed.org/series/A792RC0Q052SBEA
- population: https://fred.stlouisfed.org/series/POPTHM
- unemployment_rate: https://fred.stlouisfed.org/series/UNRATE
- consumer_confidence: https://fred.stlouisfed.org/series/UMCSENT
- personal_consumption: https://fred.stlouisfed.org/series/PCECC96
- housing_starts: https://fred.stlouisfed.org/series/HOUST
- building_permits: https://fred.stlouisfed.org/series/PERMIT
- new_home_sales: https://fred.stlouisfed.org/series/HSN1F
- existing_home_sales: https://fred.stlouisfed.org/series/EXHOSLUSM495S
- housing_inventory: https://fred.stlouisfed.org/series/MSACSR
- mortgage_rate: https://fred.stlouisfed.org/series/MORTGAGE30US
- treasury_10y: https://fred.stlouisfed.org/series/GS10
- fed_funds: https://fred.stlouisfed.org/series/FEDFUNDS
- term_spread: https://fred.stlouisfed.org/series/T10Y2Y
- m2_money_supply: https://fred.stlouisfed.org/series/M2SL
- cpi_all_items: https://fred.stlouisfed.org/series/CPIAUCSL
- cpi_less_shelter: https://fred.stlouisfed.org/series/CUSR0000SA0L2
- ppi_construction: https://fred.stlouisfed.org/series/WPUSI012011
- sp500: https://fred.stlouisfed.org/series/SP500

### 7.4 Change Log
- Initial finalized GB/XGB ensemble champion selected on 2026-03-31.
- Expanded champion search from core econometric families to broad ML, deep-learning, hybrid, and ensemble candidates.
- Added rigorous full-20 variable-selection framework for exogenous and system families.
- Promoted final champion from standalone Gradient Boosting to refined fixed-weight GB/XGB ensemble.

### 7.5 Data Dictionary

| key | series_id | transform | group | expected_sign |
| --- | --- | --- | --- | --- |
| real_gdp | GDPC1 | log_diff | demand | positive |
| per_capita_income | A792RC0Q052SBEA | log_diff | demand | positive |
| population | POPTHM | log_diff | demand | positive |
| unemployment_rate | UNRATE | level | demand | negative |
| consumer_confidence | UMCSENT | level | demand | positive |
| personal_consumption | PCECC96 | log_diff | demand | positive |
| housing_starts | HOUST | log_diff | supply | either |
| building_permits | PERMIT | log_diff | supply | either |
| new_home_sales | HSN1F | log_level | supply | either |
| housing_inventory | MSACSR | level | supply | negative |
| mortgage_rate | MORTGAGE30US | level | financing | negative |
| treasury_10y | GS10 | level | financing | negative |
| fed_funds | FEDFUNDS | level | financing | negative |
| term_spread | T10Y2Y | level | financing | positive |
| m2_money_supply | M2SL | log_diff | financing | positive |
| cpi_all_items | CPIAUCSL | log_diff | prices | either |
| cpi_less_shelter | CUSR0000SA0L2 | log_diff | prices | either |
| ppi_construction | WPUSI012011 | log_diff | prices | either |

Excluded variables:

| key | series_id | reason | first_valid_date |
| --- | --- | --- | --- |
| existing_home_sales | EXHOSLUSM495S | insufficient_history_for_primary_search_window | 2025-03-31 |
| sp500 | SP500 | insufficient_history_for_primary_search_window | 2016-06-30 |
