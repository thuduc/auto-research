# DESIGN: Autonomous HPI Forecasting Research

## 1. Purpose

This document translates the PRD into an implementation-driving design for an autonomous home price forecasting research system.

The system follows the `autoresearch` pattern:

- `prepare.py` is frozen infrastructure
- `train.py` is the mutable research surface
- `program.md` is the autonomous agent instruction set

The governed production champion may come from econometric, machine learning, deep learning, or hybrid / ensemble model classes, provided the selected model satisfies the project's reproducibility, leakage-control, and documentation requirements.

Within econometrics, the design must support both multivariate system models and single-equation dynamic / error-correction models. A complete econometric search therefore includes families such as `ARDL`, `UECM / ECM`, `DOLS / FMOLS`, `VECM`, `VAR`, `BVAR`, regime-switching models, and benchmark forecasting models such as `ETS`.

This design assumes:

- Python 3.12
- pandas as the standard dataframe and tabular workflow layer
- one isolated experiment folder per model-class experiment
- one local Python 3.12 virtual environment per experiment folder
- single-agent sequential execution as the first implementation target

## Current Reference Outcome

The current reference champion implementation is a finalized ensemble rather than a single model:

- finalized artifact: `experiments/20260331_ensemble_gb_xgb_finalize/`
- model class: `Model Ensemble`
- method: static weighted average
- weights: `Gradient Boosting 0.60`, `XGBoost 0.40`
- GOF composite: `0.6422114864199716`

The best single-model challenger remains `Gradient Boosting` at `0.6375482073353181`.

Design implication: the production finalization layer must support champion artifacts for ensembles, not only single fitted estimators. Component model provenance, fixed combination logic, and frozen weights are part of the governed artifact.

## 2. Design Goals

The design must satisfy these practical goals:

1. Make each model-class experiment fully autonomous.
2. Keep all artifacts for each experiment isolated and reproducible.
3. Preserve strict separation between frozen data preparation and mutable model research.
4. Support explainable econometric model development and post-run documentation.
5. Make the first implementation slice small enough to complete end to end.
6. Keep model-family coverage broad enough that omitted standard benchmark families are explicit exceptions rather than accidental gaps.

## 3. Non-Goals

This first design does not optimize for:

- distributed execution
- real-time dashboards
- full FHFA model-development-document generation
- vintage macroeconomic data reconstruction
- distributed GPU training

## 4. System Shape

At the highest level, the system has five layers:

1. Data preparation layer
2. Experiment isolation layer
3. Trial execution and scoring layer
4. Experiment orchestration layer
5. Champion finalization and reporting layer

The implementation is intentionally small. The agent primarily interacts with:

- `prepare.py`
- `train.py`
- `program.md`
- experiment-local folders under `experiments/`
- root summary files `results.tsv` and `leaderboard.tsv`

## 5. Core Concepts

### 5.1 Experiment

An experiment is one autonomous research loop for one model class.

Examples:

- `20260325_arimax`
- `20260325_vecm`
- `20260325_state_space`

Each experiment:

- owns its own folder
- owns its own Python 3.12 virtual environment
- owns its own local trial logs
- runs multiple trials without human intervention
- outputs one best trial for cross-class comparison

### 5.2 Trial

A trial is one concrete specification tested within an experiment.

A trial fixes:

- variable subset
- transforms or lag structure actually used by the model
- model hyperparameters
- class-specific structural options
- random seed

Depending on model family, a trial may define:

- an exogenous regressor subset
- a jointly endogenous variable system
- a feature set composed of target lags, transformed regressors, and derived inputs

Each trial produces one machine-readable artifact bundle.

### 5.3 Champion

The champion is the best governed model across all champion-eligible experiments after:

- per-class search
- cross-class refinement
- hybrid and ensemble evaluation

## 6. Directory Design

### 6.1 Root Layout

```text
auto-research-hpf/
├── PRD.md
├── DESIGN.md
├── prepare.py
├── train.py
├── program.md
├── results.tsv
├── leaderboard.tsv
├── requirements/
├── experiments/
├── data/
├── config/
├── models/
└── output/
```

### 6.2 Experiment Layout

```text
experiments/<experiment_id>/
├── venv/
├── runs/
│   └── <trial_id>/
│       ├── spec.json
│       ├── train_snapshot.py
│       ├── metrics.json
│       ├── validation_predictions.parquet
│       ├── forecast_120q.csv
│       ├── stdout.log
│       ├── stderr.log
│       └── model/
├── results.tsv
├── leaderboard.tsv
└── notes.md
```

### 6.3 Why Per-Experiment Isolation

Per-experiment folders are mandatory because they:

- prevent artifact collisions across classes
- make each model-class search independently reproducible
- keep experiment-local dependency state visible
- support later parallelization without redesign
- make cleanup and archival straightforward

## 7. Component Design

### 7.1 `prepare.py`

`prepare.py` is frozen after initial implementation.

Responsibilities:

- fetch historical FRED data
- generate structural break dummies
- align all data to quarterly frequency
- apply canonical transforms from `variables.yaml`
- build `search_panel.parquet`
- build `holdout_panel.parquet`
- build `dataset_manifest.json`
- expose evaluation utilities shared by `train.py`

`prepare.py` must not contain class-specific model logic.

Expected outputs:

- `data/processed/search_panel.parquet`
- `data/processed/holdout_panel.parquet`
- `data/processed/dataset_manifest.json`

### 7.2 `train.py`

`train.py` is the research surface and the only file the agent is expected to mutate during autonomous search.

Responsibilities:

- load prepared data
- instantiate one model specification for the active trial
- fit the model
- perform walk-forward validation
- compute GOF and diagnostics
- write trial artifacts
- optionally refit and finalize a champion

`train.py` must be able to host heterogeneous model adapters spanning:

- single-equation exogenous-regressor models
- multivariate endogenous-system models
- feature-matrix machine-learning models
- sequence-based deep-learning models
- hybrid and ensemble compositions

The internal implementation should be organized around a model-class adapter pattern even if it stays in one file.

Suggested internal structure:

- CLI parsing
- experiment context loading
- model-class dispatch
- feature construction
- fit / predict
- scoring
- artifact writing

### 7.3 `program.md`

`program.md` is the research-organization layer for the autonomous agent.

Responsibilities:

- instruct the agent how to choose the next trial
- enforce autonomy and stopping rules
- enforce experiment-folder discipline
- define search heuristics
- define troubleshooting behavior

`program.md` should evolve separately from the statistical code.

## 8. Data Layer Design

### 8.1 Historical Data Source Policy

Historical external data comes from FRED only.

Locally generated data includes:

- structural dummies
- lags
- spreads
- rolling statistics
- transformed features derived from raw FRED series

### 8.2 pandas as the Standard Data Layer

pandas is the canonical dataframe layer for:

- quarterly alignment
- feature construction
- train / validation / holdout slicing
- logging tables
- serialization to parquet or TSV

Rules:

- all prepared panels should load into pandas DataFrames
- all feature engineering should begin with pandas objects
- artifact schema generation should use pandas-compatible column naming

### 8.3 Dataset Contracts

`search_panel.parquet` must contain:

- `date`
- `hpi`
- `hpi_logdiff`
- all retained exogenous columns
- structural break dummies

`holdout_panel.parquet` must contain the same schema as `search_panel.parquet`.

`dataset_manifest.json` must contain:

- data start and end dates
- holdout boundaries
- backtest origins
- variable metadata
- transform metadata
- canonical feature names

## 9. Experiment Lifecycle

### 9.1 Experiment Bootstrap

For each model class:

1. create `experiments/<experiment_id>/`
2. create `experiments/<experiment_id>/venv/`
3. install dependencies for the required scope
4. initialize local `results.tsv` and `leaderboard.tsv`
5. seed the first trial with a simple baseline specification

Econometric completeness requires the bootstrap inventory to include both:

- system cointegration families such as `VECM`
- single-equation cointegration families such as `ARDL` and `UECM / ECM`

### 9.2 Trial Loop

Each experiment iterates through trials until the trial budget is exhausted or failure rules trigger early exit.

Per trial:

1. choose next candidate specification
2. edit `train.py`
3. snapshot inputs into the trial folder
4. activate experiment-local venv
5. run `train.py --mode search`
6. read `metrics.json`
7. update local and global logs
8. keep or discard the code change based on experiment-best GOF

### 9.3 Keep / Discard Logic

The current accepted code state for an experiment is the best trial seen so far.

If new trial GOF improves experiment-best:

- accept the trial
- commit the `train.py` state
- update experiment-local leaderboard
- update root leaderboard

If it does not improve:

- keep artifacts
- log the trial
- revert `train.py` to the last accepted experiment state

### 9.4 Failure Handling

For a failed trial:

- preserve stdout and stderr
- preserve `spec.json`
- log failure status and summary
- allow up to 2 retries for repair

If 5 consecutive trials fail inside one experiment:

- stop the experiment
- mark the experiment degraded
- continue to the next model class

## 10. Trial Selection Strategy

### 10.1 Search Heuristic

Each experiment should follow a bounded local search strategy instead of random search only.

Recommended search phases inside each experiment:

1. baseline trial
2. variable subset expansion
3. variable subset pruning
4. lag-order search
5. hyperparameter refinement
6. structural variation search

### 10.2 Variable Search Policy

Variable search should dominate hyperparameter tuning.

Suggested policy:

- 60% of trials: variable inclusion or exclusion
- 20% of trials: lag structure changes
- 20% of trials: hyperparameter or structural tuning

Variable search tactics by class:

- forward selection from economically motivated seed variables
- backward elimination from strong current trials
- channel-balanced subsets to avoid overloading one economic theme
- avoid highly collinear variable bundles unless class-specific logic supports them

### 10.3 Class-Specific Search Dimensions

Examples:

- `ARIMAX`: variable subset, lagged exogenous terms, `(p,d,q)`
- `VECM`: variable subset, lag order, cointegration rank
- `VAR`: variable subset, lag order, deterministic trend
- `State-Space`: trend type, cycle inclusion, exogenous subset
- `Dynamic Factor`: factor count, factor order, loading set
- `BVAR`: lag order, prior type, prior tightness
- `Threshold VAR / SETAR`: threshold variable, delay, regime count

## 11. Scoring Design

### 11.1 Primary Selection Metric

Selection is driven by `GOF_composite`.

The score combines:

- in-sample fit
- near-horizon validation
- far-horizon validation
- diagnostics

### 11.2 Selection Hierarchy

When two trials are close, use this tie-break order:

1. higher `GOF_composite`
2. higher `GOF_validation_near`
3. lower Theil’s U
4. fewer parameters
5. simpler economic story

This tie-break rule is important for governed model selection.

### 11.3 Plausibility and Governance Filters

A trial is ineligible if:

- forecast plausibility checks fail
- diagnostics fail catastrophically
- model behavior cannot be documented clearly

This matters most for complex ML, deep learning, and hybrid / ensemble candidates.

## 12. Logging and Artifact Design

### 12.1 Root-Level Logs

`results.tsv` is the cross-experiment trial ledger.

Each row represents one trial.

`leaderboard.tsv` is the cross-experiment summary of best results.

Each row represents one experiment-best result or approved ensemble result.

### 12.2 Experiment-Local Logs

Each experiment also maintains:

- `experiments/<experiment_id>/results.tsv`
- `experiments/<experiment_id>/leaderboard.tsv`

These local logs are the primary per-experiment record and should be append-only.

### 12.3 Trial Artifact Schema

`spec.json` should include:

- `experiment_id`
- `trial_id`
- `model_class`
- `champion_eligible`
- selected variables
- lag structure
- hyperparameters
- random seed
- dependency scope
- timestamp

`metrics.json` should include:

- all scalar GOF components
- diagnostic outcomes
- parameter count
- status
- error summary

`validation_predictions.parquet` should include:

- `experiment_id`
- `trial_id`
- `origin_date`
- `forecast_date`
- `horizon_q`
- `y_true`
- `y_pred`

## 13. Environment and Dependency Design

### 13.1 Dependency Scopes

Dependency scopes are:

- `econometric`
- `garch`
- `ensemble`
- `challenger`
- `deep_learning`

Each experiment maps to one scope for its local venv seeding.

### 13.2 Why Local venvs Per Experiment

Even if two experiments use the same dependency scope, they still get separate venvs because:

- experiment state must be isolated
- package drift must be diagnosable per experiment
- later branch-parallelization becomes simpler
- archived experiments remain self-contained

### 13.3 Stack Baseline

Mandatory stack:

- Python 3.12
- pandas

Primary governed modeling libraries:

- statsmodels
- scipy
- arch

Additional model-family support:

- scikit-learn
- xgboost
- lightgbm
- torch
- neuralprophet

## 14. Model Adapter Design

`train.py` should implement a lightweight adapter abstraction internally.

Suggested adapter interface:

- `build_features(df, spec) -> FeatureBundle`
- `fit(train_bundle, spec) -> FittedModel`
- `predict(fitted_model, forecast_context) -> pd.Series or pd.DataFrame`
- `diagnostics(fitted_model, residuals, spec) -> dict`
- `serialize(fitted_model, output_dir) -> None`

This allows one CLI and one artifact contract across all model classes.

## 15. Ensemble Design

Hybrid and ensemble methods are champion-eligible when all component models are reproducible and the combination logic is documented clearly enough for review.

Allowed hybrid / ensemble methods:

- simple averaging
- inverse-RMSE weighting
- Bayesian model averaging
- stacking on champion-eligible predictions
- econometric + ML residual correction

Inputs:

- top-5 experiment-best trial artifacts
- their `validation_predictions.parquet`

Outputs:

- one ensemble result row in `leaderboard.tsv`
- one ensemble trial-style artifact folder under a dedicated ensemble experiment

If governance conservatism increases later, the least explainable hybrid or ensemble methods should be removed first.

## 16. Parallelization Design

The first implementation should be sequential.

Parallel mode should reuse the same experiment-folder contract. Each agent simply owns a disjoint set of experiment folders.

That means parallelization does not change:

- trial schema
- local logging
- trial artifact layout
- experiment-local venv rules

The coordinator only merges root summaries.

## 17. MVP Implementation Plan

### 17.1 MVP Goal

Build one governed experiment end to end for `ARIMAX`.

The MVP is complete when the system can:

1. prepare the quarterly panel
2. create `experiments/<experiment_id>/`
3. create the experiment-local Python 3.12 venv
4. run multiple ARIMAX trials autonomously
5. score and log each trial
6. keep the best ARIMAX trial
7. finalize one 120-quarter forecast

### 17.2 MVP Build Order

1. Implement `prepare.py`
2. Implement basic `train.py` CLI and artifact writing
3. Implement ARIMAX adapter
4. Implement scoring and diagnostics
5. Implement experiment bootstrap and local/global logging
6. Implement variable search heuristics for ARIMAX
7. Implement champion finalization path
8. Add VECM and VAR
9. Add remaining econometric classes
10. Add ensembles

### 17.3 First Controlled Scope

The first implementation pass should not try to build all classes at once.

Recommended initial governed classes:

- `ARIMAX`
- `VAR`
- `VECM`

Then add:

- `State-Space`
- `ARIMAX-GARCH`
- `Dynamic Factor`
- `BVAR`
- `Threshold VAR / SETAR`
- `Ridge / Lasso / Elastic Net`
- `Random Forest`
- `Gradient Boosting`
- `Support Vector Regression`
- `LSTM / GRU`
- `TCN`
- `Transformer`
- `N-BEATS`
- `Neural Prophet`
- `VECM + ML residual correction`
- `Stacking ensemble`
- `Bayesian Model Averaging`
- `Simple averaging`

## 18. Open Decisions

These are implementation decisions that may still need tightening during coding, but they do not block starting `DESIGN.md`-driven implementation:

1. Whether ridge stacking remains champion-eligible.
2. Whether `BVAR` is implemented with `statsmodels` support or a custom lightweight implementation.
3. Whether experiment-local `notes.md` is mandatory or optional.
4. Whether root-level log merging is done by code or by agent append discipline in the first version.

## 19. Recommendation

Proceed to implementation from this design in two steps:

1. scaffold the framework and one complete `ARIMAX` experiment
2. expand class coverage once the experiment / trial lifecycle is proven stable

That is the smallest path that preserves the `autoresearch` essence while staying governed and explainable.
