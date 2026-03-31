# PRD: Autonomous Home Price Index Forecasting Research

## 1. Overview

An autonomous research system that discovers the best-fitting model for quarterly Home Price Index (HPI) forecasting over a 30-year horizon. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), the system launches one autonomous research experiment per model class, lets each experiment search over variable subsets and class-specific hyperparameters, and selects the champion model — all without human intervention.

The core loop: **propose → estimate → evaluate → keep or discard → repeat**.

## Current Research Outcome

As of 2026-03-31, the current governed champion candidate is a finalized two-model ensemble:

- `20260331_ensemble_gb_xgb_finalize / 20260331_ensemble_gb_xgb_finalize_001`
- model class: `Model Ensemble`
- method: static weighted average
- weights: `0.60 * Gradient Boosting + 0.40 * XGBoost`
- GOF composite: `0.6422114864199716`

The finalized component trials are:

- `Gradient Boosting`: `20260330_gradient_boosting_full20_300_091`
- `XGBoost`: `20260330_xgboost_full20_300_068`

Key research conclusions so far:

- the rigorous full-20 variable-selection framework materially improved the strongest ML families
- `consumer_confidence` emerged as the dominant exogenous driver for the best tree and transformer-class models
- `housing_inventory` materially improved `N-BEATS`, whose refined best reached `0.5855872642341936`
- the best econometric family remained `VECM`, but it did not approach the top ML / ensemble frontier

Current post-refinement comparison snapshot:

- `experiments/20260331_post_refinement_snapshot/leaderboard.tsv`
- `experiments/20260331_post_refinement_snapshot/results.tsv`

Terminology used in this PRD:

- **Experiment**: one autonomous autoresearch loop for a single model class
- **Trial**: one concrete specification evaluated within an experiment

---

## 2. Objectives

| Priority | Objective |
|---|---|
| **P0** | Maximize goodness-of-fit (GOF) on historical data (in-sample + pseudo-out-of-sample) |
| **P0** | Produce accurate quarterly forecasts for years 1–3 |
| **P0** | Ensure champion model methodology is explainable and defensible under FHFA-style econometric model governance |
| **P1** | Produce plausible forecasts through the full 30-year horizon |
| **P1** | Fully autonomous — no human decisions during the research loop |
| **P2** | Transparent experiment and trial log for post-hoc review |

---

## 3. System Architecture

```
auto-research-hpf/
│
├── prepare.py              # FROZEN: data download, preprocessing, evaluation
├── train.py                # MUTABLE: model definition, estimation, forecasting
├── program.md              # Agent operating instructions
├── results.tsv             # Global trial-level log across all experiments (untracked)
├── leaderboard.tsv         # Best result per model-class experiment (untracked)
│
├── requirements/           # Model-scope pip requirements used to seed experiment-local venvs
│   ├── econometric.txt     # statsmodels, scipy, pandas, etc.
│   ├── garch.txt           # arch, statsmodels, pandas, etc.
│   ├── ensemble.txt        # packages needed for econometric ensembles
│   └── challenger.txt      # optional ML/deep-learning package set for expanded search
│
├── experiments/            # One isolated folder per model-class experiment (gitignored)
│   └── <experiment_id>/
│       ├── venv/           # Python 3.12 virtual environment dedicated to this experiment
│       ├── runs/           # Trial artifacts for this experiment
│       ├── results.tsv     # Trial-level log for this experiment only
│       ├── leaderboard.tsv # Best trial summary for this experiment only
│       └── notes.md        # Optional agent scratchpad / experiment notes
│
├── data/                   # Downloaded & cached data (gitignored)
│   ├── raw/                # Raw FRED/FHFA downloads
│   └── processed/          # Cleaned quarterly panel
│
├── models/                 # Serialized champion models (gitignored)
│   └── champion/           # Current best model artifacts
│
├── config/
│   └── variables.yaml      # Variable universe definition
│
└── output/                 # Cross-experiment forecast outputs & diagnostics (gitignored)
    ├── forecasts/
    └── diagnostics/
```

### 3.1 Separation of Concerns (Following Autoresearch)

| Component | Mutability | Purpose |
|---|---|---|
| `prepare.py` | **FROZEN** (read-only) | Data acquisition, preprocessing, train/test splits, GOF evaluation functions. The agent must NEVER modify this file. |
| `train.py` | **MUTABLE** | Model specification, estimation, hyperparameter configuration, forecasting logic. This is the agent's canvas. |
| `program.md` | **Read-only** | Agent instructions — the "skill document" governing autonomous behavior. |
| `variables.yaml` | **FROZEN** | Defines the universe of candidate input variables with FRED series IDs and transformations. |

### 3.2 Experiment Isolation

Each model-class experiment must run in its own dedicated folder under `experiments/<experiment_id>/`.

- Each experiment folder owns its own Python 3.12 virtual environment at `experiments/<experiment_id>/venv/`
- Each experiment folder owns its own trial artifacts under `experiments/<experiment_id>/runs/<trial_id>/`
- Each experiment folder maintains its own local `results.tsv` and `leaderboard.tsv` in addition to the global root-level summaries
- This isolation is required for traceability, reproducibility, and cleanup of autonomous research work without mixing artifacts across experiments

---

## 4. Data Specification

### 4.1 Dependent Variable

| Variable | Source | FRED Series | FRED Location | Frequency | History |
|---|---|---|---|---|---|
| FHFA All-Transactions HPI (National) | FHFA via FRED | `USSTHPI` | `https://fred.stlouisfed.org/series/USSTHPI` | Quarterly | 1975Q1–present (~50 years) |

The national-level index is the primary target. The system uses the most recent ~30 years (1995Q1–present) as the primary estimation window, with earlier data available for robustness checks.

### 4.2 Candidate Input Variables (Exogenous)

The agent selects from this universe. Variables are organized by economic channel.

All FRED URLs in this section were live-verified on 2026-03-25 before design work proceeded.

**Demand-Side:**

| Variable | FRED Series | FRED Location | Transform | Rationale |
|---|---|---|---|---|
| Real GDP | `GDPC1` | `https://fred.stlouisfed.org/series/GDPC1` | log-difference | Aggregate demand proxy |
| Per Capita Personal Income | `A792RC0Q052SBEA` | `https://fred.stlouisfed.org/series/A792RC0Q052SBEA` | log-difference | Household purchasing power |
| Population | `POPTHM` | `https://fred.stlouisfed.org/series/POPTHM` | log-difference | Demographic demand driver |
| Unemployment Rate | `UNRATE` | `https://fred.stlouisfed.org/series/UNRATE` | level | Labor market slack |
| Consumer Confidence | `UMCSENT` | `https://fred.stlouisfed.org/series/UMCSENT` | level | Sentiment-driven demand |
| Personal Consumption Expenditure | `PCECC96` | `https://fred.stlouisfed.org/series/PCECC96` | log-difference | Consumer spending trends |

**Supply-Side:**

| Variable | FRED Series | FRED Location | Transform | Rationale |
|---|---|---|---|---|
| Housing Starts | `HOUST` | `https://fred.stlouisfed.org/series/HOUST` | log-level or growth | New supply pipeline |
| Building Permits | `PERMIT` | `https://fred.stlouisfed.org/series/PERMIT` | log-level or growth | Leading supply indicator |
| New Home Sales | `HSN1F` | `https://fred.stlouisfed.org/series/HSN1F` | log-level | Absorption rate |
| Existing Home Sales | `EXHOSLUSM495S` | `https://fred.stlouisfed.org/series/EXHOSLUSM495S` | log-level | Market liquidity |
| Housing Inventory (months supply) | `MSACSR` | `https://fred.stlouisfed.org/series/MSACSR` | level | Supply-demand balance |

**Financing/Monetary:**

| Variable | FRED Series | FRED Location | Transform | Rationale |
|---|---|---|---|---|
| 30-Year Fixed Mortgage Rate | `MORTGAGE30US` | `https://fred.stlouisfed.org/series/MORTGAGE30US` | level | Affordability constraint |
| 10-Year Treasury Yield | `GS10` | `https://fred.stlouisfed.org/series/GS10` | level | Long-term rate benchmark |
| Federal Funds Rate | `FEDFUNDS` | `https://fred.stlouisfed.org/series/FEDFUNDS` | level | Monetary policy stance |
| Term Spread (10Y – 2Y) | `T10Y2Y` | `https://fred.stlouisfed.org/series/T10Y2Y` | level | Yield curve signal |
| M2 Money Supply | `M2SL` | `https://fred.stlouisfed.org/series/M2SL` | log-difference | Liquidity proxy |

**Prices/Costs:**

| Variable | FRED Series | FRED Location | Transform | Rationale |
|---|---|---|---|---|
| CPI All Items | `CPIAUCSL` | `https://fred.stlouisfed.org/series/CPIAUCSL` | log-difference | General inflation |
| CPI Less Shelter | `CUSR0000SA0L2` | `https://fred.stlouisfed.org/series/CUSR0000SA0L2` | log-difference | Non-housing inflation (avoids circularity) |
| Producer Price Index: Construction | `WPUSI012011` | `https://fred.stlouisfed.org/series/WPUSI012011` | log-difference | Building cost pressure |
| S&P 500 | `SP500` | `https://fred.stlouisfed.org/series/SP500` | log-difference | Wealth effect / financial conditions |

**Structural Break Dummies:**

| Variable | Definition | Rationale |
|---|---|---|
| GFC step dummy | 1 from 2008Q4 onward | Structural break from financial crisis |
| COVID pulse dummy | 1 in 2020Q2 only | Transitory COVID shock |

### 4.3 Variable Substitution Groups

Variables within the same substitution group are highly correlated and measure overlapping economic signals. The agent should include **at most one member per group** in any single specification unless the post-estimation VIF for every included variable remains below 10.

| Group | Members | Rationale |
|---|---|---|
| Long rate | `MORTGAGE30US`, `GS10` | Near-collinear interest rate measures |
| Short rate / monetary policy | `FEDFUNDS`, `T10Y2Y` | Monetary policy stance proxies |
| Income / aggregate demand | `A792RC0Q052SBEA`, `GDPC1`, `PCECC96` | Correlated household demand measures |
| Supply flow | `HOUST`, `PERMIT` | Leading vs coincident new-supply indicators |
| Sales volume | `HSN1F`, `EXHOSLUSM495S` | New vs existing home sales |
| Inflation | `CPIAUCSL`, `CUSR0000SA0L2`, `WPUSI012011` | General vs sector-specific price indices |

Including multiple members from the same group is permitted but penalized: the VIF diagnostic check (Section 6.4) will score 0 if any regressor exceeds VIF = 10 in the estimated specification.

### 4.4 Data Sourcing Strategy

The system uses **FRED as the sole external historical data source** for the dependent variable and all candidate macro, housing, financing, and price drivers listed above. FRED is treated as the canonical retrieval layer even when the underlying publisher is FHFA, BEA, BLS, Census, or the University of Michigan.

The sourcing policy is:

1. **Historical target series**:
   - Fetch `USSTHPI` from FRED for the full available history
   - Use 1995Q1 onward as the primary model search window
   - Retain pre-1995 history for robustness checks, long-run averages, and feature construction when needed
2. **Historical exogenous series**:
   - Fetch all candidate drivers from their verified FRED locations only
   - Convert all retained inputs to a single quarterly frequency before modeling
   - Exclude any candidate variable that does not provide usable coverage for the full search window beginning 1995Q1
3. **Locally generated inputs**:
   - Structural break dummies such as `GFC` and `COVID` are generated directly from date rules and are not downloaded
   - Derived features such as lagged values, spreads, rolling statistics, and transformed variants are created locally from downloaded raw series
4. **Validation-time data discipline**:
   - At every backtest origin, features and transforms may only use information available as of that origin
   - No transformation may use future observations from later quarters
5. **30-year forecast inputs**:
   - Future exogenous values are not fetched from external sources during forecasting
   - System models with jointly modeled drivers may use their own model-implied paths
   - Externally regressed models must use the deterministic baseline exogenous scenario defined in Section 11.1

This policy keeps the data pipeline coherent: one external historical source, locally generated structural and derived inputs, and a separate explicit rule for future exogenous scenarios.

### 4.5 Data Preprocessing (in `prepare.py`)

1. **Download**: Fetch all series from FRED API
2. **Quarterly conversion**: Convert any higher-frequency series to quarterly using the arithmetic mean of observations within each calendar quarter; structural break dummies are generated directly at quarterly frequency
3. **Alignment**: Align all series to common quarter-end dates; no forward-fill is used across missing quarters
4. **Stationarity Transforms**: Apply the canonical transforms defined in `variables.yaml`. For the initial frozen variable set, use: `HOUST` = log-difference, `PERMIT` = log-difference, `HSN1F` = log-level, `EXHOSLUSM495S` = log-level, all other variables as specified in the tables above
5. **Missing Data**: Trim to common sample start; interpolate isolated interior gaps via linear interpolation (flag if >2 consecutive missing)
5b. **Collinearity Diagnostics**: Compute pairwise Pearson correlation matrix and Variance Inflation Factor (VIF) for each variable against all others on the transformed search panel. Export to `data/processed/collinearity_report.json` containing: `correlation_matrix`, `vif_scores` (keyed by variable name), and `high_correlation_pairs` (all pairs with |corr| > 0.85). Flag individual variables with VIF > 10.
6. **Search/Holdout split**:
   - **Search sample**: Earliest available quarter through `T_search = T−4`
   - **Holdout sample**: `T−3` through `T` (4 quarters total, never exposed during Phases 2–4)
7. **Backtest window inside search sample**:
   - **Backtest origins**: From `BACKTEST_FIRST_ORIGIN = 2005Q4` through `T_search−40Q`, advancing by 4 quarters
   - **Near horizons**: 1–12 quarters ahead
   - **Far horizons**: 13–40 quarters ahead
8. **Export**:
   - `data/processed/search_panel.parquet`: all rows through `T_search`
   - `data/processed/holdout_panel.parquet`: final 4 quarters only
   - `data/processed/dataset_manifest.json`: split boundaries, backtest origins, variable metadata, and canonical feature names

---

## 5. Model Classes to Explore

The agent must explore the following **champion-eligible** model classes. All champion-eligible models must be documented to a level appropriate for governance review, including feature rationale, estimation or training methodology, diagnostics, forecast limitations, and explainability controls proportionate to the model family.

### 5.1 Econometric / Time-Series Models

For completeness, the econometric search must cover both:

- multivariate system models where `hpi` and selected macro series are jointly endogenous
- single-equation dynamic models where `hpi` is the primary dependent variable and other series enter as exogenous or distributed-lag drivers

The champion search is not considered econometrically complete unless it includes both `VECM`-style system error-correction models and single-equation `ECM`-style alternatives.

| Model Class | Library | Key Hyperparameters |
|---|---|---|
| **ARDL** (Autoregressive Distributed Lag) | `statsmodels` | Target lag order, per-regressor lag orders, variable subset, deterministic terms |
| **UECM / ECM** (Error Correction Model) | `statsmodels` | Error-correction form derived from ARDL or cointegrating equation, lag orders, variable subset |
| **DOLS / FMOLS** (Cointegration regression benchmarks) | `statsmodels` or custom | Lead/lag count for differenced regressors, variable subset, deterministic terms |
| **VECM** (Vector Error Correction) | `statsmodels` | Lag order (1–8), cointegration rank (Johansen), variable subset |
| **VAR** (Vector Autoregression) | `statsmodels` | Lag order, variable subset, trend specification |
| **ARIMAX** | `statsmodels` | (p,d,q) orders, exogenous variable subset |
| **ARIMAX-GARCH** | `arch` | ARIMAX orders + GARCH(p,q) or GJR-GARCH |
| **State-Space / Unobserved Components** | `statsmodels` | Trend type (local level, local linear, random walk), cycle, exogenous |
| **Dynamic Factor Model** | `statsmodels` | Number of factors, factor order, variable loadings |
| **Bayesian VAR (BVAR)** | `statsmodels` or custom | Prior type (Minnesota, Normal-Wishart), tightness, lag order |
| **Threshold VAR / SETAR** | custom or `statsmodels` | Threshold variable, delay, regime count |
| **ETS / Exponential Smoothing** | `statsmodels` | Error/trend specification, damping, initialization |
| **Markov-Switching AR / ARX** | `statsmodels` | Regime count, switching intercepts/slopes, lag order, exogenous subset |

### 5.2 Machine Learning Models

| Model Class | Library | Key Hyperparameters |
|---|---|---|
| **Ridge / Lasso / Elastic Net** | `scikit-learn` | Alpha, L1 ratio, lag features, polynomial features |
| **Random Forest** | `scikit-learn` | n_estimators, max_depth, lag features |
| **Gradient Boosting (XGBoost / LightGBM)** | `xgboost` / `lightgbm` | Learning rate, depth, n_estimators, regularization |
| **Support Vector Regression** | `scikit-learn` | Kernel, C, epsilon, gamma |

### 5.3 Deep Learning Models

| Model Class | Library | Key Hyperparameters |
|---|---|---|
| **LSTM / GRU** | `torch` | Hidden size, layer count, dropout, lookback window |
| **Temporal Convolutional Network (TCN)** | `torch` | Channel widths, kernel size, dilation schedule, dropout |
| **Transformer (time-series)** | `torch` | d_model, attention heads, encoder depth, lookback window |
| **N-BEATS** | `torch` | Stack type, block depth, hidden width, lookback window |
| **Neural Prophet** | `neuralprophet` | n_lags, trend settings, seasonality settings, learning rate |

### 5.4 Hybrid / Ensemble Models

| Model Class | Approach |
|---|---|
| **VECM + ML residual correction** | Fit an econometric base model, then model residual structure with an ML learner |
| **Simple averaging** | Equal-weight average of top-K champion-eligible models |
| **Inverse-RMSE weighting** | Weighted average of top-K champion-eligible models using validation RMSE |
| **Bayesian Model Averaging** | Weighted combination based on posterior model probabilities |
| **Stacking ensemble** | Meta-learner built from validation predictions of champion-eligible base models |

---

## 6. Goodness-of-Fit (GOF) Evaluation Framework

### 6.1 Primary Scoring Metric

The **composite GOF score** determines whether a trial is kept or discarded within its model-class experiment:

```
GOF_composite = 0.40 × GOF_insample
              + 0.35 × GOF_validation_near
              + 0.15 × GOF_validation_far
              + 0.10 × GOF_diagnostic
```

Where:

| Component | Metric | Description |
|---|---|---|
| `GOF_insample` | 1 − (RMSE / RMSE_naive) | In-sample fit on the full search sample (`search_panel.parquet`) relative to a random-walk-with-drift benchmark |
| `GOF_validation_near` | 1 − (RMSE_1-12Q / RMSE_naive_1-12Q) | Walk-forward accuracy aggregated over all backtest origins and horizons 1–12 |
| `GOF_validation_far` | 1 − (RMSE_13-40Q / RMSE_naive_13-40Q) | Walk-forward accuracy aggregated over all backtest origins and horizons 13–40 |
| `GOF_diagnostic` | Diagnostic pass rate (0–1) | Fraction of statistical diagnostics passed |

Higher is better. A score of 0 means random-walk performance; 1 means perfect fit.

### 6.2 Detailed Evaluation Metrics (computed for reporting, not ranking)

**Point Forecast Accuracy:**

| Metric | Formula | Used For |
|---|---|---|
| RMSE | √(mean(e²)) | Primary accuracy measure |
| MAE | mean(\|e\|) | Robust accuracy measure |
| MAPE | mean(\|e/y\| × 100) | Scale-independent accuracy |
| Theil's U | RMSE_model / RMSE_naive | Benchmark comparison (< 1 = better than naive) |
| Directional Accuracy | % correct direction calls | Critical for qualitative validity |

**Statistical Diagnostics:**

| Diagnostic | Test | Pass Criterion |
|---|---|---|
| Residual autocorrelation | Ljung-Box (12 lags) | p > 0.05 |
| Residual normality | Jarque-Bera | p > 0.01 |
| Heteroskedasticity | ARCH-LM (4 lags) | p > 0.05 |
| Parameter stability | CUSUM | Stays within 5% bounds |
| Stationarity of residuals | ADF test | p < 0.05 |

**Forecast Plausibility Checks (hard constraints — a trial auto-fails if violated):**

| Check | Criterion |
|---|---|
| HPI never goes negative | All forecasted levels > 0 |
| No extreme quarterly jumps | \|quarterly change\| < 15% |
| 30-year cumulative appreciation | Between −20% and +500% (annualized ~−0.7% to ~6%) |
| Non-degenerate forecast | Standard deviation of forecast path > 0 (not flat line) |

### 6.3 Backtesting Protocol

**Expanding-window walk-forward validation on the search sample:**

1. Load `search_panel.parquet` only; holdout rows are not available during search
2. Start with initial training window ending at `BACKTEST_FIRST_ORIGIN = 2005Q4`
3. At each origin, estimate the model and forecast 40 quarters ahead
4. Aggregate errors across horizons 1–12 into `GOF_validation_near`
5. Aggregate errors across horizons 13–40 into `GOF_validation_far`
6. Advance the origin by 4 quarters
7. Stop at the last origin that still leaves 40 observable quarters inside the search sample, namely `T_search−40Q`

This is the only validation protocol used for ranking models. The final 4-quarter holdout is reserved exclusively for the champion report in Phase 5.

### 6.4 Diagnostic Score Construction

`GOF_diagnostic` is the mean of the following binary checks, each scored as `1` for pass and `0` for fail:

- Ljung-Box
- Jarque-Bera
- ARCH-LM
- CUSUM
- ADF on residuals
- Economic sign check for coefficients that have an expected sign in linear models
- VIF check: all included regressors have VIF < 10 in the estimated specification (using `statsmodels.stats.outliers_influence.variance_inflation_factor`)
- Forecast plausibility checks

For nonparametric models where coefficient signs are not defined, the economic sign check is omitted from the denominator. For models without explicit regressors (e.g., univariate ARIMA), the VIF check is omitted from the denominator.

---

## 7. Autonomous Research Loop

### 7.1 Phase 1: Initialization (runs once)

```
1. Create git repo, branch: autoresearch/<date_tag>
2. Create one folder per model-class experiment under `experiments/<experiment_id>/`
3. Create a Python 3.12 virtual environment inside each experiment folder:
   python3.12 -m venv experiments/<experiment_id>/venv
4. Install experiment dependencies into the local venv:
    source experiments/<experiment_id>/venv/bin/activate && pip install -r requirements/<scope>.txt
5. Run prepare.py to download and preprocess all data
6. Verify data integrity (no NaN in critical columns, sufficient history)
7. Estimate naive benchmark (random walk with drift)
8. Record benchmark GOF scores in the global and per-experiment `results.tsv`
9. Initialize train.py with the simplest model (ARIMAX(1,1,0) with mortgage rate)
```

### 7.2 Phase 2: Per-Class Autonomous Experiments

The agent launches one autonomous experiment per champion-eligible model class. Each experiment runs a bounded number of trials that search over variable subsets, hyperparameters, and class-specific structural choices. The overall scheduler explores classes **breadth-first**, spending a controlled number of trials per experiment before moving on, then returns to promising experiments.

```
FOR each champion-eligible model class in [
    VECM,
    VAR,
    ARIMAX,
    ARIMAX-GARCH,
    State-Space,
    Dynamic Factor,
    BVAR,
    Threshold VAR,
    Ridge/Lasso/Elastic Net,
    Random Forest,
    Gradient Boosting,
    Support Vector Regression,
    LSTM/GRU,
    TCN,
    Transformer,
    N-BEATS,
    Neural Prophet,
    VECM + ML residual correction,
    Simple Averaging,
    Bayesian Model Averaging,
    Stacking Ensemble
]:
    SET EXPERIMENT_ID = <date_tag>_<model_class>
    SET trials_this_experiment = 0
    SET best_gof_this_experiment = -inf
    SET ENV_SCOPE according to model class:
        - econometric: VECM, VAR, ARIMAX, State-Space, Dynamic Factor, BVAR, Threshold VAR
        - garch: ARIMAX-GARCH
        - challenger: Ridge/Lasso/Elastic Net, Random Forest, Gradient Boosting, Support Vector Regression
        - deep_learning: LSTM/GRU, TCN, Transformer, N-BEATS, Neural Prophet
        - ensemble: VECM + ML residual correction, Simple Averaging, Bayesian Model Averaging, Stacking Ensemble
    SET EXPERIMENT_DIR = experiments/<EXPERIMENT_ID>/

    WHILE trials_this_experiment < MAX_TRIALS_PER_EXPERIMENT (default: 15):
        1. Read current train.py, global results.tsv, global leaderboard.tsv, and experiment-local logs in <EXPERIMENT_DIR>
        2. Allocate a unique TRIAL_ID and output directory: <EXPERIMENT_DIR>/runs/<TRIAL_ID>/
        3. Select next trial (variable subset, lag structure, hyperparameters, architecture tweak)
        4. Edit train.py with the new specification
        5. Save the exact trial inputs:
           - copy of train.py to <EXPERIMENT_DIR>/runs/<TRIAL_ID>/train_snapshot.py
           - trial metadata to <EXPERIMENT_DIR>/runs/<TRIAL_ID>/spec.json, including EXPERIMENT_ID and TRIAL_ID
        6. Activate the experiment-local Python 3.12 venv: source <EXPERIMENT_DIR>/venv/bin/activate
        7. Run: python train.py --mode search --model-class <class> --experiment-id <EXPERIMENT_ID> --trial-id <TRIAL_ID> --output-dir <EXPERIMENT_DIR>/runs/<TRIAL_ID>
        8. Read <EXPERIMENT_DIR>/runs/<TRIAL_ID>/metrics.json
        9. IF crash: diagnose from <EXPERIMENT_DIR>/runs/<TRIAL_ID>/stderr.log, attempt fix, max 2 retries, then skip
        10. IF GOF_composite improved over experiment-best:
               COMMIT train.py, update best_gof_this_experiment
            ELSE:
               RESTORE train.py to the last accepted version without deleting <EXPERIMENT_DIR>/runs/<TRIAL_ID>/
        11. Record trial result in both <EXPERIMENT_DIR>/results.tsv and root-level results.tsv using metrics.json, EXPERIMENT_ID, and TRIAL_ID
        12. trials_this_experiment += 1

    Record experiment-best to both <EXPERIMENT_DIR>/leaderboard.tsv and root-level leaderboard.tsv
```

### 7.3 Phase 3: Cross-Class Refinement

After all champion-eligible classes have been explored:

```
1. Rank model classes by best GOF_composite
2. Return to top-3 classes for deeper search:
   - Fine-grained hyperparameter sweeps
   - Variable selection refinement (forward/backward stepwise within the model)
   - Architecture variations within the leading model families (for example VECM cointegration rank, State-Space trend/cycle specification, gradient boosting depth/regularization, or neural lookback/width choices)
   3. Additional MAX_REFINEMENT_TRIALS (default: 20) per top class experiment
   4. Update leaderboard.tsv
```

### 7.4 Phase 4: Hybrid / Ensemble Construction

```
1. Take top-5 individual models from leaderboard
2. Try ensemble methods:
    - Simple average
    - Inverse-RMSE weighted average
    - Stacking ensemble on champion-eligible validation predictions
    - Bayesian Model Averaging (if applicable)
    - Residual-correction hybrids where supported
3. Build hybrid and ensemble candidates from the persisted `validation_predictions.parquet` artifacts of the top-performing trials; no ensemble may rely on ad hoc re-runs
4. Evaluate each hybrid and ensemble via same GOF framework
5. If any hybrid or ensemble beats best individual: it becomes champion
6. Record hybrid and ensemble results to leaderboard.tsv
```

### 7.5 Optional Benchmark Extensions

```
1. Optional: run ablation studies, alternative feature sets, or external benchmark models after the champion-eligible search is complete
2. Record benchmark-only results to results.tsv and leaderboard.tsv with `champion_eligible = false`
3. Benchmark-only models may be compared against the production champion for research purposes
4. Benchmark-only models may not replace the champion unless explicitly promoted into the governed class list above
```

### 7.6 Phase 5: Champion Finalization

```
1. Select champion model (highest GOF_composite)
2. Re-estimate champion on the full search sample (`search_panel.parquet`) for production forecasts
3. Generate 30-year (120-quarter) forecast under each of the three exogenous scenarios (Baseline, Adverse, Severely Adverse) as defined in Section 11.1.1:
   - For system models: Baseline uses model-implied path; Adverse/Severely Adverse use conditional forecasting or shocked initial conditions
   - For externally regressed models: feed each scenario's exogenous path directly
   - Verify plausibility checks (Section 6.2) pass under Baseline; log but do not auto-fail Adverse/Severely Adverse plausibility violations
4. Run final holdout evaluation on `holdout_panel.parquet` and report it separately; do not use it for selection
5. Serialize model to models/champion/
6. Export forecasts to output/forecasts/champion_forecast.csv (all three scenarios)
7. Generate diagnostic report to output/diagnostics/champion_report.json
```

---

## 8. Agent Operating Protocol (`program.md`)

### 8.1 Core Rules

1. **Never modify `prepare.py` or `variables.yaml`.** These are frozen infrastructure.
2. **Never stop to ask the human.** If stuck, try an alternative approach. If stuck 3 times on the same model class, move to the next class.
3. **All trials must complete within 10 minutes.** Kill anything exceeding this.
4. **Git tracks accepted trial states.** Only trials that improve the experiment-best score are committed to the search branch.
5. **Machine-readable trial artifacts track every trial.** Every trial, including crashes and discarded candidates, must persist under `experiments/<experiment_id>/runs/<TRIAL_ID>/`.
6. **Always record results** to `results.tsv` even for crashes and failures.
7. **Respect the holdout.** Never look at or use `holdout_panel.parquet` during search.

### 8.2 Experiment Strategy Guidelines

- **Start simple.** Begin each model class with the simplest specification (fewest variables, default hyperparameters).
- **Vary one thing at a time** when debugging, but allow multi-variable changes when exploring.
- **Variable selection matters more than tuning.** Spend 60% of trials on variable subsets, 40% on hyperparameters.
- **Economic coherence matters.** If a model's coefficients have wrong signs (e.g., higher mortgage rates → higher HPI), note this in the log and penalize via the diagnostic score.
- **Respect substitution groups (Section 4.3).** When building variable subsets, prefer at most one variable per substitution group. If including multiple members from the same group, verify post-estimation VIF < 10 for all regressors. If VIF exceeds 10, drop the weaker contributor (lower t-statistic or less economically motivated variable).
- **Keep system dimension manageable.** For VAR/VECM with endogenous variables, limit the system to 5–6 endogenous variables to avoid parameter explosion (a VAR(4) with 6 endogenous variables has ~150 parameters on ~120 observations).
- **Parsimony is a virtue.** Given equal GOF, prefer the simpler model (fewer parameters). Record parameter count.
- **Champion eligibility requires explainability.** Any model that cannot be clearly documented in FHFA-style model development terms is ineligible for champion selection.
- **If a model class consistently crashes**, move on after 5 failed consecutive trials within that experiment.

### 8.3 Logging Format

Each trial appends to `results.tsv`:

```
experiment_id  trial_id  accepted_commit  model_class  champion_eligible  gof_composite  gof_insample  gof_val_near  gof_val_far  gof_diag  rmse  mae  theil_u  n_params  status  description  artifact_dir  error_summary
```

Each experiment-level result appends to `leaderboard.tsv`:

```
rank  experiment_id  model_class  champion_eligible  best_trial_id  best_commit  gof_composite  gof_insample  gof_val_near  gof_val_far  rmse_1yr  rmse_3yr  n_params  n_trials  description
```

### 8.4 Experiment Artifact Contract

Each trial directory `experiments/<experiment_id>/runs/<TRIAL_ID>/` must contain:

- `spec.json`: experiment and trial metadata, selected variables, hyperparameters, random seed, and model class
- `train_snapshot.py`: exact `train.py` used for the trial
- `metrics.json`: all scalar metrics required for `results.tsv`
- `validation_predictions.parquet`: one row per origin-horizon prediction with columns `experiment_id`, `trial_id`, `origin_date`, `forecast_date`, `horizon_q`, `y_true`, `y_pred`
- `forecast_120q.csv`: 120-quarter production forecast generated from the fitted run configuration
- `model/`: serialized fitted object(s), if serialization is supported by the model class
- `stdout.log` and `stderr.log`: raw execution logs

---

## 9. Technical Requirements

### 9.1 Technical Stack and Dependencies

The mandatory technical foundation for this project is:

- **Python 3.12**
- **pandas**

```
# Core
python3.12              # Exact version required
numpy
pandas                  # Primary dataframe and panel-manipulation library
scipy
statsmodels             # VECM, VAR, ARIMAX, State-Space, Dynamic Factor
arch                    # GARCH family
scikit-learn            # ML models and stacking ensembles
xgboost                 # Gradient boosting models
lightgbm                # Gradient boosting models
torch                   # Deep learning models
neuralprophet           # NeuralProphet model class

# Data
fredapi                 # FRED API client
pyarrow                 # Parquet I/O

# Utilities
pyyaml                  # Config parsing
tabulate                # Results display
matplotlib              # Diagnostic plots (optional)
```

### 9.2 Environment

- **Python 3.12** — all agents must use `python3.12` to create virtual environments and run scripts
- **Local virtual environment per experiment** — every model-class experiment gets its own isolated Python 3.12 virtual environment inside its own experiment folder. This is mandatory for traceability and isolation. Example layout:
  ```
  experiments/
  └── <experiment_id>/
      ├── venv/           # Python 3.12 environment for this experiment only
      ├── runs/           # Trial artifacts for this experiment
      ├── results.tsv
      └── leaderboard.tsv
  ```
  Each experiment creates its venv on first use:
  ```bash
  python3.12 -m venv experiments/<experiment_id>/venv
  source experiments/<experiment_id>/venv/bin/activate
  pip install -r requirements/<scope>.txt
  ```
- **pandas-centered data handling** — all tabular preprocessing, split handling, feature generation, and experiment logging should use pandas as the standard dataframe layer
- **Single machine** — no distributed training needed (data is small: ~120 quarterly observations × ~25 variables)
- **GPU optional** — econometric and classical ML experiments can run on CPU; deep learning experiments may use GPU when available but must still run on a single machine
- **FRED API key** required — set as environment variable `FRED_API_KEY=157a7cf9abe3230e37c951c320455324`

### 9.3 Runtime Estimates

| Phase | Estimated Experiments | Time per Experiment | Total Time |
|---|---|---|---|
| Initialization | 1 | 2 min | 2 min |
| Per-Class Experiments (17 individual model classes × 60 trials) | 1,020 trials | 1–10 min | 17–170 hours |
| Cross-Class Refinement (5 experiments × 20 additional trials) | 100 trials | 1–10 min | 2–17 hours |
| Hybrid / Ensemble Construction | 10–20 | 1–5 min | 10–100 min |
| Optional Benchmark Extensions | 0–20 | 1–10 min | 0–3 hours |
| Champion Finalization | 1 | 5 min | 5 min |
| **Total** | **~1,131–1,141** | | **~19–190 hours** |

### 9.4 Script Interface Contract

`prepare.py` is a frozen batch job with no required CLI arguments:

```bash
python prepare.py
```

`train.py` is mutable internally but must expose a stable command-line interface:

```bash
python train.py \
  --mode {search,finalize} \
  --model-class <class> \
  --experiment-id <EXPERIMENT_ID> \
  --trial-id <TRIAL_ID> \
  --output-dir experiments/<EXPERIMENT_ID>/runs/<TRIAL_ID>
```

Behavior by mode:

- `--mode search`: loads `search_panel.parquet`, computes in-sample metrics, walk-forward validation metrics, diagnostics, and writes all required trial artifacts under the experiment folder
- `--mode finalize`: loads `search_panel.parquet` for refit, generates the 120-quarter forecast, optionally evaluates against `holdout_panel.parquet`, and writes champion-ready artifacts

`metrics.json` must include at least:

```json
{
  "experiment_id": "20260325_arimax",
  "trial_id": "20260325_arimax_001",
  "model_class": "ARIMAX",
  "champion_eligible": true,
  "description": "ARIMAX(1,1,0) with mortgage rate",
  "gof_composite": 0.61,
  "gof_insample": 0.57,
  "gof_validation_near": 0.66,
  "gof_validation_far": 0.49,
  "gof_diagnostic": 0.86,
  "rmse": 1.42,
  "mae": 1.11,
  "theil_u": 0.82,
  "directional_accuracy": 0.74,
  "n_params": 8,
  "status": "ok",
  "diagnostics_passed": ["ljung_box", "arch_lm", "adf"],
  "diagnostics_failed": ["jarque_bera"],
  "error_summary": ""
}
```

---

## 10. Output Deliverables

### 10.1 Champion Forecast

File: `output/forecasts/champion_forecast.csv`

| Column | Description |
|---|---|
| `date` | Quarterly date (YYYY-QN) |
| `scenario` | `baseline`, `adverse`, or `severely_adverse` |
| `hpi_actual` | Historical HPI (where available) |
| `hpi_forecast` | Point forecast |
| `hpi_lower_90` | 90% prediction interval lower bound |
| `hpi_upper_90` | 90% prediction interval upper bound |
| `hpi_lower_50` | 50% prediction interval lower bound |
| `hpi_upper_50` | 50% prediction interval upper bound |

### 10.2 Diagnostic Report

File: `output/diagnostics/champion_report.json`

```json
{
  "champion": {
    "model_class": "VECM",
    "description": "VECM(4) with mortgage_rate, income, housing_starts, unemployment",
    "commit": "abc1234",
    "n_params": 42,
    "gof_composite": 0.847,
    "gof_insample": 0.912,
    "gof_validation_near": 0.823,
    "gof_validation_far": 0.714,
    "gof_diagnostic": 0.875,
    "rmse_1yr": 1.23,
    "rmse_3yr": 2.45,
    "theil_u": 0.67,
    "directional_accuracy": 0.82,
    "diagnostics_passed": ["ljung_box", "adf", "cusum"],
    "diagnostics_failed": ["jarque_bera"],
    "scenario_summary": {
      "baseline_30yr_cumulative": 0.89,
      "adverse_30yr_cumulative": -0.12,
      "severely_adverse_30yr_cumulative": -0.34,
      "baseline_annualized": 0.021,
      "adverse_annualized": -0.004,
      "severely_adverse_annualized": -0.014
    }
  },
  "runner_up": { ... },
  "experiment_summary": {
    "total_experiments": 8,
    "total_trials": 192,
    "total_runtime_hours": 6.3,
    "classes_explored": 8,
    "best_per_class": { ... }
  }
}
```

### 10.3 Experiment Log

File: `results.tsv` — full history of every trial attempted within each model-class experiment, including crashes and discards.

### 10.4 Leaderboard

File: `leaderboard.tsv` — ranked summary of the best result from each model-class experiment plus ensembles.

### 10.5 Per-Trial Prediction Artifacts

File: `experiments/<EXPERIMENT_ID>/runs/<TRIAL_ID>/validation_predictions.parquet`

| Column | Description |
|---|---|
| `experiment_id` | Model-class experiment identifier |
| `trial_id` | Unique trial identifier within the experiment |
| `origin_date` | Walk-forward estimation origin |
| `forecast_date` | Predicted quarter |
| `horizon_q` | Forecast horizon in quarters |
| `y_true` | Observed HPI value |
| `y_pred` | Model prediction |

These files are required inputs for the ensemble phase.

---

## 11. Constraints and Guardrails

### 11.1 Data Leakage Prevention

- The holdout set (most recent 4 quarters) is **never loaded** during Phases 2–4
- `prepare.py` enforces the split by writing separate `search_panel.parquet` and `holdout_panel.parquet` files
- In `--mode search`, `train.py` may only read `search_panel.parquet`
- Future exogenous variable values for the 30-year forecast use the following default policy:
  - For system models that jointly model the target and drivers internally (VAR, VECM, BVAR, Dynamic Factor), use the model-implied endogenous forecast path under **Baseline**; use conditional forecasting or shocked initial conditions under **Adverse** and **Severely Adverse** (see Section 11.1.1)
  - For externally regressed models, feed each scenario's deterministic exogenous path directly (see Section 11.1.1)
- Prediction intervals are produced as follows:
  - if the model class provides analytical intervals, use them
  - otherwise bootstrap 1,000 forecast paths using quarter-block residual resampling (block size = 4) from search-sample backtest residuals
  - report the 5th/95th and 25th/75th percentiles as the 90% and 50% bands
- Any champion-eligible ML, deep learning, or hybrid model may be selected as champion if it satisfies the same leakage, reproducibility, and documentation requirements as the econometric classes

#### 11.1.1 Exogenous Forecast Scenarios

The system produces forecasts under three named scenarios. All scenarios are deterministic and mechanically derived from search-sample history, requiring no external scenario provider.

**Stress calibration inputs** (computed in `prepare.py` and exported to `data/processed/dataset_manifest.json`):

For each candidate variable, compute from the search sample:
- `long_run_mean`: mean of the trailing 80 quarters (20 years)
- `long_run_std`: standard deviation of the trailing 80 quarters
- `last_observed`: value at `T_search`

**Scenario definitions for externally regressed models:**

| Scenario | Rate/Level Variables | Growth-Rate Variables | Dummies |
|---|---|---|---|
| **Baseline** | Last observed → hold constant 8Q, then linearly mean-revert to `long_run_mean` by Q20, then constant | Trailing 20Q mean | GFC=1, COVID=0 |
| **Adverse** | Last observed → linear shift to `long_run_mean + 1σ` stress by Q8, hold through Q20, then mean-revert to `long_run_mean` by Q40 | Trailing 20Q mean − 1σ | GFC=1, COVID=0 |
| **Severely Adverse** | Last observed → immediate 2σ shock at Q1, hold through Q12, then slow mean-revert to `long_run_mean` by Q40 | Trailing 20Q mean − 2σ | GFC=1, COVID=0 |

Where σ = `long_run_std` for each variable.

**Scenario definitions for system models (VAR, VECM, BVAR, Dynamic Factor):**

| Scenario | Approach |
|---|---|
| **Baseline** | Model-implied endogenous forecast path (unchanged) |
| **Adverse** | Apply the Adverse exogenous stress paths as conditional forecasts if the model supports conditioning, otherwise shock initial conditions by +1σ for rate variables and −1σ for growth variables |
| **Severely Adverse** | Apply the Severely Adverse exogenous stress paths as conditional forecasts if supported, otherwise shock initial conditions by +2σ for rate variables and −2σ for growth variables |

The scenario framework produces 3 × 120 = 360 forecast rows. Plausibility checks (Section 6.2) are enforced under **Baseline**; violations under Adverse and Severely Adverse are logged but do not auto-fail the trial.

### 11.2 Overfitting Mitigation

- GOF_composite weights validation (50%) more than in-sample (40%)
- Walk-forward backtesting with multiple origins prevents lucky-split overfitting
- Parameter count is logged; models with >100 parameters relative to ~120 observations will be flagged
- Champion selection favors parsimonious, interpretable specifications over marginally better but less defensible alternatives

### 11.3 Reproducibility

- Random seeds are fixed in `prepare.py` (seed=42)
- Every accepted trial is a git commit with the exact `train.py` that produced it
- Every trial, including discarded or crashed trials, is reproducible from `experiments/<EXPERIMENT_ID>/runs/<TRIAL_ID>/train_snapshot.py` plus `spec.json`
- `results.tsv` links experiment IDs, trial IDs, and accepted commits to metrics

---

## 12. Agent Deployment Options

### 12.1 Single Agent (Recommended Start)

One Claude Code agent runs the full loop sequentially. Simple, no coordination overhead.

```
claude --agent program.md
```

### 12.2 Agent Farm (Parallel Exploration)

Multiple agents may explore different model-family groups simultaneously on separate git branches, with a coordinator agent merging results. Optional benchmark extensions, if enabled, run separately and do not participate in champion selection.

```
Coordinator Agent
├── Agent-A: branch autoresearch/econometric_a   → VECM, VAR, ARIMAX, ARIMAX-GARCH
├── Agent-B: branch autoresearch/econometric_b   → State-Space, Dynamic Factor, BVAR, Threshold VAR
├── Agent-C: branch autoresearch/ml              → Ridge/Lasso/Elastic Net, Random Forest, Gradient Boosting, SVR
├── Agent-D: branch autoresearch/deep_learning   → LSTM/GRU, TCN, Transformer, N-BEATS, Neural Prophet
└── Agent-E: branch autoresearch/hybrid_ensemble → waits for A-D, then builds hybrids and ensembles
```

Coordination protocol:
- Each agent writes to experiment-local `experiments/<experiment_id>/results.tsv` files
- Coordinator polls for completion, merges leaderboards, launches Agent-D
- Agent-D reads all leaderboard files and builds ensembles from top individual models

### 12.3 Scaling Knobs

| Parameter | Default | Description |
|---|---|---|
| `MAX_TRIALS_PER_EXPERIMENT` | 15 | Trials during each model-class experiment's breadth-first search |
| `MAX_REFINEMENT_TRIALS` | 20 | Additional trials for each of the top-3 class experiments |
| `MAX_TRIAL_TIME_SECONDS` | 600 | Kill trials exceeding this |
| `TOP_K_CLASSES` | 3 | Number of classes for refinement phase |
| `TOP_K_ENSEMBLE` | 5 | Number of models for ensemble construction |
| `BACKTEST_FIRST_ORIGIN` | 2005Q4 | First expanding-window origin |
| `BACKTEST_STEP_QUARTERS` | 4 | Quarters between backtest origins |

---

## 13. Success Criteria

| Criterion | Target |
|---|---|
| Champion GOF_composite | > 0.70 (meaningfully better than random walk) |
| Theil's U (1-year horizon) | < 0.80 |
| Theil's U (3-year horizon) | < 0.90 |
| Directional accuracy | > 75% |
| Champion model class | Any listed champion-eligible class |
| Champion model documentation readiness | Full variable rationale, estimation logic, diagnostics, and limitations are explainable |
| Governed champion-search runtime | < 16 hours |
| Trials completed | > 150 |
| Model classes explored | >= 6 |
| Zero human intervention during run | Yes |
| Scenario spread reasonableness | Adverse < Baseline at every horizon; Severely Adverse shows meaningful additional stress beyond Adverse |
| Scenario monotonicity | At every forecast horizon: severely_adverse ≤ adverse ≤ baseline |

---

## 14. Post-Run Human Review Checklist

After the autonomous run completes, the human should review:

1. **`leaderboard.tsv`** — Are the top models sensible? Do the best classes align with economic intuition?
2. **Champion coefficient signs** — Do they make economic sense? (e.g., higher rates → lower HPI)
3. **Forecast plausibility** — Plot the 30-year forecast. Does the trajectory look reasonable?
4. **Holdout performance** — Check the holdout results in `champion_report.json`. Large degradation from validation → holdout signals overfitting.
5. **Git log** — Review the sequence of successful accepted trials to understand the optimization trajectory.
6. **Crash rate** — If >30% of trials crashed, investigate common failure modes for future improvement.
