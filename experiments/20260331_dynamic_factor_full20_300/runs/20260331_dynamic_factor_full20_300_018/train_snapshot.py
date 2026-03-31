#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from arch.unitroot.cointegration import DynamicOLS, FullyModifiedOLS
from scipy.stats import jarque_bera, norm, t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank


ROOT = Path(__file__).resolve().parent
SEARCH_PANEL_PATH = ROOT / "data" / "processed" / "search_panel.parquet"
HOLDOUT_PANEL_PATH = ROOT / "data" / "processed" / "holdout_panel.parquet"
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.json"
VARIABLES_PATH = ROOT / "config" / "variables.yaml"
ROOT_RESULTS_PATH = ROOT / "results.tsv"
ROOT_LEADERBOARD_PATH = ROOT / "leaderboard.tsv"

RESULTS_HEADER = [
    "experiment_id",
    "trial_id",
    "accepted_commit",
    "model_class",
    "champion_eligible",
    "gof_composite",
    "gof_insample",
    "gof_val_near",
    "gof_val_far",
    "gof_diag",
    "rmse",
    "mae",
    "theil_u",
    "n_params",
    "status",
    "description",
    "artifact_dir",
    "error_summary",
]

LEADERBOARD_HEADER = [
    "rank",
    "experiment_id",
    "model_class",
    "champion_eligible",
    "best_trial_id",
    "best_commit",
    "gof_composite",
    "gof_insample",
    "gof_val_near",
    "gof_val_far",
    "rmse_1yr",
    "rmse_3yr",
    "n_params",
    "n_trials",
    "description",
]

DEFAULT_EXPERIMENT_SPEC: dict[str, Any] = {
    "sample_start": "1995-03-31",
    "target_column": "hpi",
    "target_date_column": "date",
    "champion_eligible": True,
    "description": "ARIMAX(1,1,0) with mortgage_rate",
    "arimax": {
        "order": [1, 1, 0],
        "trend": "t",
        "exogenous": ["mortgage_rate"],
        "include_intercept": False,
        "enforce_stationarity": False,
        "enforce_invertibility": False,
        "maxiter": 200,
    },
}
ACTIVE_EXPERIMENT_SPEC: dict[str, Any] = copy.deepcopy(DEFAULT_EXPERIMENT_SPEC)


class ArimaxGarchFit:
    def __init__(self, mean_result: Any, volatility_result: Any, residual_scale: float, distribution: str) -> None:
        self.mean_result = mean_result
        self.volatility_result = volatility_result
        self.residual_scale = float(residual_scale)
        self.distribution = distribution
        mean_params = model_param_lookup(mean_result)
        vol_params = model_param_lookup(volatility_result)
        self.params = pd.Series({**mean_params, **vol_params}, dtype=float)


class BVarFit:
    def __init__(
        self,
        coefficients: np.ndarray,
        intercept: np.ndarray,
        sigma_u: np.ndarray,
        lags: int,
        trend: str,
        endogenous: list[str],
        fittedvalues: pd.DataFrame,
        residuals: pd.DataFrame,
        train_frame: pd.DataFrame,
        prior_type: str,
        tightness: float,
    ) -> None:
        self.coefficients = coefficients
        self.intercept = intercept
        self.sigma_u = sigma_u
        self.lags = lags
        self.trend = trend
        self.endogenous = endogenous
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.train_frame = train_frame
        self.prior_type = prior_type
        self.tightness = tightness
        self.k_ar = lags
        self.params = pd.Series(self._flatten_params(), dtype=float)

    def _flatten_params(self) -> dict[str, float]:
        params: dict[str, float] = {}
        for eq_idx, equation in enumerate(self.endogenous):
            if self.trend == "c":
                params[f"{equation}_const"] = float(self.intercept[eq_idx])
            for lag in range(self.lags):
                for var_idx, variable in enumerate(self.endogenous):
                    params[f"{equation}_L{lag + 1}_{variable}"] = float(self.coefficients[lag, eq_idx, var_idx])
        params["prior_tightness"] = float(self.tightness)
        return params


class ThresholdVarFit:
    def __init__(
        self,
        low_coefficients: np.ndarray,
        high_coefficients: np.ndarray,
        low_intercept: np.ndarray,
        high_intercept: np.ndarray,
        sigma_u: np.ndarray,
        lags: int,
        trend: str,
        endogenous: list[str],
        threshold_variable: str,
        threshold_value: float,
        delay: int,
        fittedvalues: pd.DataFrame,
        residuals: pd.DataFrame,
        train_frame: pd.DataFrame,
        threshold_quantile: float,
        regime_counts: dict[str, int],
    ) -> None:
        self.low_coefficients = low_coefficients
        self.high_coefficients = high_coefficients
        self.low_intercept = low_intercept
        self.high_intercept = high_intercept
        self.sigma_u = sigma_u
        self.lags = lags
        self.trend = trend
        self.endogenous = endogenous
        self.threshold_variable = threshold_variable
        self.threshold_value = float(threshold_value)
        self.delay = delay
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.train_frame = train_frame
        self.threshold_quantile = float(threshold_quantile)
        self.regime_counts = regime_counts
        self.k_ar = lags
        self.params = pd.Series(self._flatten_params(), dtype=float)

    def _flatten_params(self) -> dict[str, float]:
        params: dict[str, float] = {
            "threshold_value": self.threshold_value,
            "threshold_quantile": self.threshold_quantile,
            "delay": float(self.delay),
        }
        for regime_name, intercept, coefficients in [
            ("low", self.low_intercept, self.low_coefficients),
            ("high", self.high_intercept, self.high_coefficients),
        ]:
            for eq_idx, equation in enumerate(self.endogenous):
                if self.trend == "c":
                    params[f"{regime_name}_{equation}_const"] = float(intercept[eq_idx])
                for lag in range(self.lags):
                    for var_idx, variable in enumerate(self.endogenous):
                        params[f"{regime_name}_{equation}_L{lag + 1}_{variable}"] = float(coefficients[lag, eq_idx, var_idx])
        return params


class RegularizedLinearFit:
    def __init__(
        self,
        estimator: Any,
        scaler: Any,
        feature_names: list[str],
        target_lags: list[int],
        exogenous: list[str],
        fittedvalues: pd.Series,
        residuals: pd.Series,
        params: dict[str, float],
    ) -> None:
        self.estimator = estimator
        self.scaler = scaler
        self.feature_names = feature_names
        self.target_lags = target_lags
        self.exogenous = exogenous
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.params = pd.Series(params, dtype=float)
        self.residual_std = max(float(residuals.std(ddof=0)), 1e-6) if len(residuals) else 1e-6


class TabularMlFit:
    def __init__(
        self,
        estimator: Any,
        feature_names: list[str],
        target_lags: list[int],
        exogenous: list[str],
        fittedvalues: pd.Series,
        residuals: pd.Series,
        params: dict[str, float],
        scaler: Any | None = None,
    ) -> None:
        self.estimator = estimator
        self.scaler = scaler
        self.feature_names = feature_names
        self.target_lags = target_lags
        self.exogenous = exogenous
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.params = pd.Series(params, dtype=float)
        self.residual_std = max(float(residuals.std(ddof=0)), 1e-6) if len(residuals) else 1e-6


class CointegrationRegressionFit:
    def __init__(
        self,
        estimator: Any,
        method: str,
        exogenous: list[str],
        trend: str,
        fittedvalues: pd.Series,
        residuals: pd.Series,
        params: pd.Series,
        leads: int = 0,
        lags: int = 0,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.exogenous = exogenous
        self.trend = trend
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.params = params.astype(float)
        self.leads = int(leads)
        self.lags = int(lags)
        self.residual_std = max(float(residuals.std(ddof=0)), 1e-6) if len(residuals) else 1e-6


class MarkovSwitchingFit:
    def __init__(
        self,
        estimator: Any,
        exogenous: list[str],
        params: pd.Series,
        fittedvalues: pd.Series,
        residuals: pd.Series,
        transition_matrix: np.ndarray,
        last_probabilities: np.ndarray,
        fit_strategy: str,
    ) -> None:
        self.estimator = estimator
        self.exogenous = exogenous
        self.params = params.astype(float)
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        self.last_probabilities = np.asarray(last_probabilities, dtype=float)
        self.residual_std = max(float(residuals.std(ddof=0)), 1e-6) if len(residuals) else 1e-6
        self._fit_strategy = fit_strategy


class NeuralProphetFit:
    def __init__(
        self,
        model: Any,
        target_lags: int,
        exogenous: list[str],
        train_history: pd.DataFrame,
        fittedvalues: pd.Series,
        residuals: pd.Series,
        params: dict[str, float],
    ) -> None:
        self.model = model
        self.target_lags = target_lags
        self.exogenous = exogenous
        self.train_history = train_history
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.params = pd.Series(params, dtype=float)
        self.residual_std = max(float(residuals.std(ddof=0)), 1e-6) if len(residuals) else 1e-6


class RecurrentSequenceFit:
    def __init__(
        self,
        model: Any,
        scaler_x: Any,
        scaler_y: Any,
        lookback: int,
        exogenous: list[str],
        fittedvalues: pd.Series,
        residuals: pd.Series,
        params: dict[str, float],
    ) -> None:
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.lookback = lookback
        self.exogenous = exogenous
        self.fittedvalues = fittedvalues
        self.resid = residuals
        self.params = pd.Series(params, dtype=float)
        self.residual_std = max(float(residuals.std(ddof=0)), 1e-6) if len(residuals) else 1e-6


def get_active_spec() -> dict[str, Any]:
    return ACTIVE_EXPERIMENT_SPEC


def set_active_spec(spec: dict[str, Any]) -> None:
    ACTIVE_EXPERIMENT_SPEC.clear()
    ACTIVE_EXPERIMENT_SPEC.update(copy.deepcopy(spec))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "finalize", "experiment"], required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-trials", type=int, default=300)
    parser.add_argument("--improvement-threshold", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


class TrialLogger:
    def __init__(self, output_dir: Path) -> None:
        self.stdout_path = output_dir / "stdout.log"
        self.stderr_path = output_dir / "stderr.log"

    def info(self, message: str) -> None:
        with self.stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def error(self, message: str) -> None:
        with self.stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


def ensure_parent_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = output_dir.parent.parent
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir, experiment_dir / "leaderboard.tsv"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def write_snapshot(output_dir: Path) -> None:
    snapshot_path = output_dir / "train_snapshot.py"
    snapshot_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")


def write_spec(output_dir: Path, args: argparse.Namespace, dependency_scope: str) -> None:
    payload = {
        "experiment_id": args.experiment_id,
        "trial_id": args.trial_id,
        "model_class": args.model_class,
        "champion_eligible": get_active_spec()["champion_eligible"],
        "dependency_scope": dependency_scope,
        "spec": get_active_spec(),
    }
    with (output_dir / "spec.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    search_panel = pd.read_parquet(SEARCH_PANEL_PATH)
    holdout_panel = pd.read_parquet(HOLDOUT_PANEL_PATH)
    for frame in (search_panel, holdout_panel):
        frame["date"] = pd.to_datetime(frame["date"])
    return search_panel, holdout_panel, load_json(MANIFEST_PATH), load_yaml(VARIABLES_PATH)


def resolve_dependency_scope(model_class: str) -> str:
    if model_class == "ARIMAX-GARCH":
        return "garch"
    if model_class in {"ARIMAX", "ARDL", "UECM / ECM", "DOLS / FMOLS", "ETS", "Markov-Switching AR / ARX", "VAR", "VECM", "State-Space", "Dynamic Factor", "BVAR", "Threshold VAR", "Threshold VAR / SETAR"}:
        return "econometric"
    if model_class in {"Ridge / Lasso / Elastic Net", "Random Forest", "Gradient Boosting", "Support Vector Regression", "XGBoost", "LightGBM", "NeuralProphet", "LSTM / GRU", "TCN", "N-BEATS", "Transformer"}:
        return "challenger"
    return "challenger"


def filter_sample(frame: pd.DataFrame) -> pd.DataFrame:
    sample_start = pd.Timestamp(get_active_spec()["sample_start"])
    return frame.loc[frame["date"] >= sample_start].reset_index(drop=True)


def build_exog(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame | None:
    if not columns:
        return None
    return frame[columns].astype(float)


def compute_naive_insample_rmse(y: pd.Series) -> float:
    diffs = y.diff().dropna()
    drift = diffs.mean() if not diffs.empty else 0.0
    preds = y.shift(1) + drift
    aligned = pd.DataFrame({"y": y, "pred": preds}).dropna()
    return rmse(aligned["y"], aligned["pred"])


def forecast_naive_path(train_y: pd.Series, steps: int) -> np.ndarray:
    diffs = train_y.diff().dropna()
    drift = diffs.mean() if not diffs.empty else 0.0
    start = float(train_y.iloc[-1])
    horizons = np.arange(1, steps + 1, dtype=float)
    return start + drift * horizons


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(np.square(a - p))))


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def expected_signs(variable_config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in variable_config["exogenous"]:
        sign = item.get("expected_sign")
        if sign in {"positive", "negative"}:
            mapping[item["key"]] = sign
    return mapping


def full_exogenous_universe(variable_config: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    configured = [item["key"] for item in variable_config["exogenous"]]
    return [column for column in configured if column in frame.columns]


def univariate_exogenous_screen(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
    variable_config: dict[str, Any],
    max_variables: int = 10,
) -> list[str]:
    available = full_exogenous_universe(variable_config, search_frame)
    origins = [pd.Timestamp(item) for item in manifest.get("backtest_origins", [])]
    scored: list[tuple[float, str]] = []

    for column in available:
        gains: list[float] = []
        for origin in origins:
            origin_idx = search_frame.index[search_frame["date"] == origin]
            if len(origin_idx) != 1:
                continue
            end_idx = int(origin_idx[0])
            train = search_frame.iloc[: end_idx + 1].copy()
            future = search_frame.iloc[end_idx + 1 : end_idx + 13].copy()
            if len(future) < 12:
                continue

            baseline = forecast_naive_path(train["hpi"], len(future))
            baseline_rmse = rmse(future["hpi"], baseline)

            model_frame = train[["hpi", column]].dropna().copy()
            if len(model_frame) < 24:
                continue
            correlation = float(abs(model_frame["hpi"].corr(model_frame[column])))
            if not np.isfinite(correlation):
                continue

            future_aligned = future[[column]].dropna()
            if len(future_aligned) < len(future):
                continue
            simple_pred = np.repeat(float(train["hpi"].iloc[-1]), len(future)) + correlation * (future[column].to_numpy(dtype=float) - float(train[column].iloc[-1]))
            screened_rmse = rmse(future["hpi"], simple_pred)
            gains.append(1.0 - (screened_rmse / baseline_rmse if baseline_rmse else np.inf))

        if gains:
            stability_penalty = float(np.std(gains)) if len(gains) > 1 else 0.0
            score = float(np.mean(gains) - 0.5 * stability_penalty)
            scored.append((score, column))

    scored.sort(reverse=True)
    ranked = [column for _, column in scored]
    if not ranked:
        ranked = available
    return ranked[:max_variables]


def screened_exogenous_subsets(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
    variable_config: dict[str, Any],
    *,
    include_empty: bool,
    max_ranked: int,
    max_subset_size: int,
) -> list[list[str]]:
    ranked = univariate_exogenous_screen(search_frame, manifest, variable_config, max_variables=max_ranked)
    subsets: list[list[str]] = [[]] if include_empty else []
    subsets.extend([[column] for column in ranked])

    if max_subset_size >= 2:
        for idx, first in enumerate(ranked):
            for second in ranked[idx + 1 :]:
                subsets.append([first, second])

    if max_subset_size >= 3:
        top_for_triples = ranked[: min(6, len(ranked))]
        for i, first in enumerate(top_for_triples):
            for j, second in enumerate(top_for_triples[i + 1 :], start=i + 1):
                for third in top_for_triples[j + 1 :]:
                    subsets.append([first, second, third])

    unique_sets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for subset in subsets:
        deduped = tuple(dict.fromkeys(subset))
        if deduped not in seen:
            seen.add(deduped)
            unique_sets.append(list(deduped))
    return unique_sets


def screened_system_subsets(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
    variable_config: dict[str, Any],
    *,
    max_ranked: int,
    max_subset_size: int,
) -> list[list[str]]:
    ranked = univariate_exogenous_screen(search_frame, manifest, variable_config, max_variables=max_ranked)
    subsets: list[list[str]] = []
    subsets.extend([[column] for column in ranked])

    if max_subset_size >= 2:
        for idx, first in enumerate(ranked):
            for second in ranked[idx + 1 :]:
                subsets.append([first, second])

    if max_subset_size >= 3:
        top_for_triples = ranked[: min(6, len(ranked))]
        for i, first in enumerate(top_for_triples):
            for j, second in enumerate(top_for_triples[i + 1 :], start=i + 1):
                for third in top_for_triples[j + 1 :]:
                    subsets.append([first, second, third])

    unique_endogenous: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for subset in subsets:
        retained = ["hpi"] + [column for column in subset if column in search_frame.columns and column != "hpi"]
        key = tuple(dict.fromkeys(retained))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        unique_endogenous.append(list(key))
    return unique_endogenous


def build_future_exog(search_frame: pd.DataFrame, variable_config: dict[str, Any], horizon: int) -> pd.DataFrame:
    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=horizon, freq="QE-DEC")
    base = pd.DataFrame({"date": future_dates})
    trailing_20q = search_frame.tail(min(20, len(search_frame)))
    trailing_20y = search_frame.tail(min(80, len(search_frame)))
    config_map = {item["key"]: item for item in variable_config["exogenous"] if item["key"] in search_frame.columns}

    for column, spec in config_map.items():
        history = search_frame[column].astype(float)
        last_value = float(history.iloc[-1])
        if spec["transform"] in {"log_diff"}:
            fill_value = float(trailing_20q[column].mean())
            base[column] = fill_value
        else:
            long_run = float(trailing_20y[column].mean())
            values: list[float] = []
            for quarter in range(1, horizon + 1):
                if quarter <= 8:
                    value = last_value
                elif quarter <= 20:
                    blend = (quarter - 8) / 12.0
                    value = last_value + (long_run - last_value) * blend
                else:
                    value = long_run
                values.append(value)
            base[column] = values

    base["gfc_dummy"] = 1.0
    base["covid_dummy"] = 0.0
    return base


def regularized_linear_feature_frame(frame: pd.DataFrame, target_lags: list[int], exogenous: list[str]) -> pd.DataFrame:
    if not target_lags:
        raise RuntimeError("Regularized linear models require at least one target lag.")
    features = frame[["date", "hpi"] + exogenous].copy()
    for lag in target_lags:
        features[f"hpi_lag_{lag}"] = features["hpi"].shift(lag)
    columns = ["date", "hpi"] + [f"hpi_lag_{lag}" for lag in target_lags] + exogenous
    return features[columns].dropna().reset_index(drop=True)


def tabular_feature_frame(frame: pd.DataFrame, target_lags: list[int], exogenous: list[str]) -> pd.DataFrame:
    return regularized_linear_feature_frame(frame, target_lags, exogenous)


def fit_regularized_linear(train_frame: pd.DataFrame) -> RegularizedLinearFit:
    try:
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Regularized linear models require scikit-learn in the active environment.") from exc

    spec = get_active_spec()["regularized_linear"]
    model_frame = regularized_linear_feature_frame(
        train_frame,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
    )
    feature_names = [column for column in model_frame.columns if column not in {"date", "hpi"}]
    if not feature_names:
        raise RuntimeError("Regularized linear model has no features after lag/exogenous construction.")

    x_values = model_frame[feature_names].to_numpy(dtype=float)
    y_values = model_frame["hpi"].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_values)

    model_type = spec["model_type"]
    alpha = float(spec["alpha"])
    fit_intercept = bool(spec.get("fit_intercept", True))
    if model_type == "ridge":
        estimator = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif model_type == "lasso":
        estimator = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=int(spec.get("max_iter", 5000)), random_state=42)
    elif model_type == "elastic_net":
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=float(spec["l1_ratio"]),
            fit_intercept=fit_intercept,
            max_iter=int(spec.get("max_iter", 5000)),
            random_state=42,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported regularized linear model type: {model_type}")

    estimator.fit(x_scaled, y_values)
    fitted_values = pd.Series(estimator.predict(x_scaled), index=model_frame.index, dtype=float)
    residuals = pd.Series(model_frame["hpi"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float)

    params: dict[str, float] = {}
    intercept = float(np.asarray(estimator.intercept_, dtype=float).reshape(-1)[0]) if np.size(estimator.intercept_) else float(estimator.intercept_)
    params["intercept"] = intercept
    for name, value in zip(feature_names, np.asarray(estimator.coef_, dtype=float).reshape(-1)):
        params[name] = float(value)
    params["alpha"] = alpha
    if model_type == "elastic_net":
        params["l1_ratio"] = float(spec["l1_ratio"])

    return RegularizedLinearFit(
        estimator=estimator,
        scaler=scaler,
        feature_names=feature_names,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
        fittedvalues=pd.Series(fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float),
        residuals=residuals,
        params=params,
    )


def fit_random_forest(train_frame: pd.DataFrame) -> TabularMlFit:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Random Forest requires scikit-learn in the active environment.") from exc

    spec = get_active_spec()["random_forest"]
    model_frame = tabular_feature_frame(
        train_frame,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
    )
    feature_names = [column for column in model_frame.columns if column not in {"date", "hpi"}]
    if not feature_names:
        raise RuntimeError("Random Forest has no features after lag/exogenous construction.")

    x_values = model_frame[feature_names].to_numpy(dtype=float)
    y_values = model_frame["hpi"].to_numpy(dtype=float)
    estimator = RandomForestRegressor(
        n_estimators=int(spec["n_estimators"]),
        max_depth=None if spec.get("max_depth") is None else int(spec["max_depth"]),
        min_samples_leaf=int(spec.get("min_samples_leaf", 1)),
        max_features=spec.get("max_features", 1.0),
        random_state=42,
        n_jobs=-1,
    )
    estimator.fit(x_values, y_values)
    fitted_values = pd.Series(estimator.predict(x_values), index=model_frame.index, dtype=float)
    residuals = pd.Series(model_frame["hpi"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float)

    params = {
        "n_estimators": float(spec["n_estimators"]),
        "max_depth": float(spec["max_depth"]) if spec.get("max_depth") is not None else -1.0,
        "min_samples_leaf": float(spec.get("min_samples_leaf", 1)),
    }
    for name, value in zip(feature_names, np.asarray(estimator.feature_importances_, dtype=float)):
        params[f"importance_{name}"] = float(value)

    return TabularMlFit(
        estimator=estimator,
        scaler=None,
        feature_names=feature_names,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
        fittedvalues=pd.Series(fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float),
        residuals=residuals,
        params=params,
    )


def fit_gradient_boosting(train_frame: pd.DataFrame) -> TabularMlFit:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Gradient Boosting requires scikit-learn in the active environment.") from exc

    spec = get_active_spec()["gradient_boosting"]
    model_frame = tabular_feature_frame(
        train_frame,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
    )
    feature_names = [column for column in model_frame.columns if column not in {"date", "hpi"}]
    if not feature_names:
        raise RuntimeError("Gradient Boosting has no features after lag/exogenous construction.")

    x_values = model_frame[feature_names].to_numpy(dtype=float)
    y_values = model_frame["hpi"].to_numpy(dtype=float)
    estimator = GradientBoostingRegressor(
        n_estimators=int(spec["n_estimators"]),
        learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]),
        min_samples_leaf=int(spec.get("min_samples_leaf", 1)),
        subsample=float(spec.get("subsample", 1.0)),
        random_state=42,
    )
    estimator.fit(x_values, y_values)
    fitted_values = pd.Series(estimator.predict(x_values), index=model_frame.index, dtype=float)
    residuals = pd.Series(model_frame["hpi"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float)

    params = {
        "n_estimators": float(spec["n_estimators"]),
        "learning_rate": float(spec["learning_rate"]),
        "max_depth": float(spec["max_depth"]),
        "min_samples_leaf": float(spec.get("min_samples_leaf", 1)),
        "subsample": float(spec.get("subsample", 1.0)),
    }
    for name, value in zip(feature_names, np.asarray(estimator.feature_importances_, dtype=float)):
        params[f"importance_{name}"] = float(value)

    return TabularMlFit(
        estimator=estimator,
        scaler=None,
        feature_names=feature_names,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
        fittedvalues=pd.Series(fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float),
        residuals=residuals,
        params=params,
    )


def fit_support_vector_regression(train_frame: pd.DataFrame) -> TabularMlFit:
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Support Vector Regression requires scikit-learn in the active environment.") from exc

    spec = get_active_spec()["support_vector_regression"]
    model_frame = tabular_feature_frame(
        train_frame,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
    )
    feature_names = [column for column in model_frame.columns if column not in {"date", "hpi"}]
    if not feature_names:
        raise RuntimeError("Support Vector Regression has no features after lag/exogenous construction.")

    x_values = model_frame[feature_names].to_numpy(dtype=float)
    y_values = model_frame["hpi"].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_values)
    estimator = SVR(
        kernel=str(spec["kernel"]),
        C=float(spec["c"]),
        epsilon=float(spec["epsilon"]),
        gamma=spec.get("gamma", "scale"),
    )
    estimator.fit(x_scaled, y_values)
    fitted_values = pd.Series(estimator.predict(x_scaled), index=model_frame.index, dtype=float)
    residuals = pd.Series(model_frame["hpi"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float)

    params = {
        "c": float(spec["c"]),
        "epsilon": float(spec["epsilon"]),
        "kernel_linear": 1.0 if spec["kernel"] == "linear" else 0.0,
        "kernel_rbf": 1.0 if spec["kernel"] == "rbf" else 0.0,
    }
    if spec.get("gamma") not in {None, "scale", "auto"}:
        params["gamma"] = float(spec["gamma"])

    return TabularMlFit(
        estimator=estimator,
        scaler=scaler,
        feature_names=feature_names,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
        fittedvalues=pd.Series(fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float),
        residuals=residuals,
        params=params,
    )


def fit_xgboost(train_frame: pd.DataFrame) -> TabularMlFit:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("XGBoost requires the xgboost package in the active environment.") from exc

    spec = get_active_spec()["xgboost"]
    model_frame = tabular_feature_frame(
        train_frame,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
    )
    feature_names = [column for column in model_frame.columns if column not in {"date", "hpi"}]
    if not feature_names:
        raise RuntimeError("XGBoost has no features after lag/exogenous construction.")

    x_values = model_frame[feature_names].to_numpy(dtype=float)
    y_values = model_frame["hpi"].to_numpy(dtype=float)
    estimator = XGBRegressor(
        n_estimators=int(spec["n_estimators"]),
        max_depth=int(spec["max_depth"]),
        learning_rate=float(spec["learning_rate"]),
        subsample=float(spec.get("subsample", 1.0)),
        colsample_bytree=float(spec.get("colsample_bytree", 1.0)),
        reg_lambda=float(spec.get("reg_lambda", 1.0)),
        min_child_weight=float(spec.get("min_child_weight", 1.0)),
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    estimator.fit(x_values, y_values)
    fitted_values = pd.Series(estimator.predict(x_values), index=model_frame.index, dtype=float)
    residuals = pd.Series(model_frame["hpi"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float)

    params = {
        "n_estimators": float(spec["n_estimators"]),
        "max_depth": float(spec["max_depth"]),
        "learning_rate": float(spec["learning_rate"]),
        "subsample": float(spec.get("subsample", 1.0)),
        "colsample_bytree": float(spec.get("colsample_bytree", 1.0)),
        "reg_lambda": float(spec.get("reg_lambda", 1.0)),
        "min_child_weight": float(spec.get("min_child_weight", 1.0)),
    }
    for name, value in zip(feature_names, np.asarray(estimator.feature_importances_, dtype=float)):
        params[f"importance_{name}"] = float(value)

    return TabularMlFit(
        estimator=estimator,
        scaler=None,
        feature_names=feature_names,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
        fittedvalues=pd.Series(fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float),
        residuals=residuals,
        params=params,
    )


def fit_lightgbm(train_frame: pd.DataFrame) -> TabularMlFit:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LightGBM requires the lightgbm package in the active environment.") from exc

    spec = get_active_spec()["lightgbm"]
    model_frame = tabular_feature_frame(
        train_frame,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
    )
    feature_names = [column for column in model_frame.columns if column not in {"date", "hpi"}]
    if not feature_names:
        raise RuntimeError("LightGBM has no features after lag/exogenous construction.")

    x_values = model_frame[feature_names].to_numpy(dtype=float)
    y_values = model_frame["hpi"].to_numpy(dtype=float)
    estimator = LGBMRegressor(
        n_estimators=int(spec["n_estimators"]),
        learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]),
        num_leaves=int(spec["num_leaves"]),
        min_child_samples=int(spec.get("min_child_samples", 20)),
        subsample=float(spec.get("subsample", 1.0)),
        colsample_bytree=float(spec.get("colsample_bytree", 1.0)),
        reg_lambda=float(spec.get("reg_lambda", 0.0)),
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    estimator.fit(x_values, y_values)
    fitted_values = pd.Series(estimator.predict(x_values), index=model_frame.index, dtype=float)
    residuals = pd.Series(model_frame["hpi"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float)

    params = {
        "n_estimators": float(spec["n_estimators"]),
        "learning_rate": float(spec["learning_rate"]),
        "max_depth": float(spec["max_depth"]),
        "num_leaves": float(spec["num_leaves"]),
        "min_child_samples": float(spec.get("min_child_samples", 20)),
        "subsample": float(spec.get("subsample", 1.0)),
        "colsample_bytree": float(spec.get("colsample_bytree", 1.0)),
        "reg_lambda": float(spec.get("reg_lambda", 0.0)),
    }
    for name, value in zip(feature_names, np.asarray(estimator.feature_importances_, dtype=float)):
        params[f"importance_{name}"] = float(value)

    return TabularMlFit(
        estimator=estimator,
        scaler=None,
        feature_names=feature_names,
        target_lags=[int(lag) for lag in spec["target_lags"]],
        exogenous=list(spec["exogenous"]),
        fittedvalues=pd.Series(fitted_values.to_numpy(dtype=float), index=model_frame.index, dtype=float),
        residuals=residuals,
        params=params,
    )


def neuralprophet_training_frame(frame: pd.DataFrame, exogenous: list[str]) -> pd.DataFrame:
    columns = ["date", "hpi"] + exogenous
    result = frame[columns].copy()
    result = result.rename(columns={"date": "ds", "hpi": "y"})
    result["ds"] = pd.to_datetime(result["ds"])
    return result


def fit_neuralprophet(train_frame: pd.DataFrame) -> NeuralProphetFit:
    try:
        from neuralprophet import NeuralProphet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NeuralProphet requires the neuralprophet package in the active environment.") from exc

    spec = get_active_spec()["neuralprophet"]
    exogenous = list(spec["exogenous"])
    history = neuralprophet_training_frame(train_frame, exogenous)
    n_lags = int(spec["n_lags"])
    model = NeuralProphet(
        n_lags=n_lags,
        n_forecasts=1,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        learning_rate=float(spec["learning_rate"]),
        epochs=int(spec["epochs"]),
        batch_size=int(spec["batch_size"]),
        trend_reg=float(spec.get("trend_reg", 0.0)),
    )
    for column in exogenous:
        model = model.add_future_regressor(column)

    model.fit(history, freq="QE", progress="none", minimal=True)
    forecast = model.predict(history)
    prediction_column = [column for column in forecast.columns if column.startswith("yhat")][0]
    fitted_frame = history[["ds", "y"]].merge(forecast[["ds", prediction_column]], on="ds", how="left").dropna().reset_index(drop=True)
    fitted_values = pd.Series(fitted_frame[prediction_column].to_numpy(dtype=float), index=fitted_frame.index, dtype=float)
    residuals = pd.Series(fitted_frame["y"].to_numpy(dtype=float) - fitted_values.to_numpy(dtype=float), index=fitted_frame.index, dtype=float)
    params = {
        "n_lags": float(n_lags),
        "epochs": float(spec["epochs"]),
        "learning_rate": float(spec["learning_rate"]),
        "batch_size": float(spec["batch_size"]),
        "trend_reg": float(spec.get("trend_reg", 0.0)),
        "n_exogenous": float(len(exogenous)),
    }

    return NeuralProphetFit(
        model=model,
        target_lags=n_lags,
        exogenous=exogenous,
        train_history=history,
        fittedvalues=fitted_values,
        residuals=residuals,
        params=params,
    )


def neuralprophet_future_frame(fitted: NeuralProphetFit, future_exog: pd.DataFrame) -> pd.DataFrame:
    future = fitted.model.make_future_dataframe(
        fitted.train_history,
        periods=len(future_exog),
        n_historic_predictions=False,
    )
    future = future[["ds"]].copy()
    future["y"] = np.nan
    if fitted.exogenous:
        exog = future_exog.copy().rename(columns={"date": "ds"})
        exog["ds"] = pd.to_datetime(exog["ds"])
        future = future.merge(exog, on="ds", how="left")
    return future


def neuralprophet_forecast_path(
    fitted: NeuralProphetFit,
    future_exog: pd.DataFrame,
) -> np.ndarray:
    future = neuralprophet_future_frame(fitted, future_exog)
    forecast = fitted.model.predict(future)
    prediction_column = [column for column in forecast.columns if column.startswith("yhat")][0]
    return forecast[prediction_column].to_numpy(dtype=float)


def recurrent_feature_matrix(frame: pd.DataFrame, exogenous: list[str]) -> np.ndarray:
    columns = ["hpi"] + exogenous
    return frame[columns].astype(float).to_numpy(dtype=float)


def recurrent_sequences(frame: pd.DataFrame, lookback: int, exogenous: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = recurrent_feature_matrix(frame, exogenous)
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for idx in range(lookback, len(values)):
        x_list.append(values[idx - lookback : idx])
        y_list.append(float(values[idx, 0]))
    return np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=float)


def fit_lstm_gru(train_frame: pd.DataFrame) -> RecurrentSequenceFit:
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LSTM / GRU requires torch and scikit-learn in the active environment.") from exc

    spec = get_active_spec()["lstm_gru"]
    exogenous = list(spec["exogenous"])
    lookback = int(spec["lookback"])
    x_raw, y_raw = recurrent_sequences(train_frame, lookback, exogenous)
    if len(x_raw) == 0:
        raise RuntimeError("LSTM / GRU requires more observations than the lookback window.")

    sample_count, _, feature_count = x_raw.shape
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_raw.reshape(sample_count * lookback, feature_count)).reshape(sample_count, lookback, feature_count)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)

    hidden_size = int(spec["hidden_size"])
    num_layers = int(spec["num_layers"])
    cell_type = str(spec["cell_type"])
    dropout = float(spec.get("dropout", 0.0)) if num_layers > 1 else 0.0

    class RecurrentRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_cls = torch.nn.LSTM if cell_type == "lstm" else torch.nn.GRU
            self.recurrent = recurrent_cls(
                input_size=feature_count,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs, _ = self.recurrent(inputs)
            return self.output(outputs[:, -1, :]).squeeze(-1)

    device = torch.device("cpu")
    model = RecurrentRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    model.train()
    for _ in range(int(spec["epochs"])):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        fitted_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1, 1)
    fitted_values = scaler_y.inverse_transform(fitted_scaled).reshape(-1)
    residuals = y_raw - fitted_values

    params = {
        "lookback": float(lookback),
        "hidden_size": float(hidden_size),
        "num_layers": float(num_layers),
        "epochs": float(spec["epochs"]),
        "learning_rate": float(spec["learning_rate"]),
        "dropout": float(spec.get("dropout", 0.0)),
        "is_lstm": 1.0 if cell_type == "lstm" else 0.0,
        "is_gru": 1.0 if cell_type == "gru" else 0.0,
    }

    return RecurrentSequenceFit(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        lookback=lookback,
        exogenous=exogenous,
        fittedvalues=pd.Series(fitted_values, index=pd.RangeIndex(len(fitted_values)), dtype=float),
        residuals=pd.Series(residuals, index=pd.RangeIndex(len(residuals)), dtype=float),
        params=params,
    )


def fit_tcn(train_frame: pd.DataFrame) -> RecurrentSequenceFit:
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TCN requires torch and scikit-learn in the active environment.") from exc

    spec = get_active_spec()["tcn"]
    exogenous = list(spec["exogenous"])
    lookback = int(spec["lookback"])
    x_raw, y_raw = recurrent_sequences(train_frame, lookback, exogenous)
    if len(x_raw) == 0:
        raise RuntimeError("TCN requires more observations than the lookback window.")

    sample_count, _, feature_count = x_raw.shape
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_raw.reshape(sample_count * lookback, feature_count)).reshape(sample_count, lookback, feature_count)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)

    channels = int(spec["channels"])
    num_blocks = int(spec["num_blocks"])
    kernel_size = int(spec.get("kernel_size", 2))
    dropout = float(spec.get("dropout", 0.0))

    class CausalConv1d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.left_padding > 0:
                inputs = torch.nn.functional.pad(inputs, (self.left_padding, 0))
            return self.conv(inputs)

    class TemporalBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.conv1 = CausalConv1d(in_channels, out_channels, dilation=dilation)
            self.conv2 = CausalConv1d(out_channels, out_channels, dilation=dilation)
            self.dropout = torch.nn.Dropout(dropout)
            self.downsample = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            residual = inputs if self.downsample is None else self.downsample(inputs)
            outputs = torch.relu(self.conv1(inputs))
            outputs = self.dropout(outputs)
            outputs = torch.relu(self.conv2(outputs))
            outputs = self.dropout(outputs)
            return torch.relu(outputs + residual)

    class TemporalConvRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[torch.nn.Module] = []
            in_channels = feature_count
            for block_idx in range(num_blocks):
                blocks.append(TemporalBlock(in_channels, channels, dilation=2**block_idx))
                in_channels = channels
            self.network = torch.nn.Sequential(*blocks)
            self.output = torch.nn.Linear(channels, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.network(inputs.transpose(1, 2))
            return self.output(outputs[:, :, -1]).squeeze(-1)

    device = torch.device("cpu")
    model = TemporalConvRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    model.train()
    for _ in range(int(spec["epochs"])):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        fitted_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1, 1)
    fitted_values = scaler_y.inverse_transform(fitted_scaled).reshape(-1)
    residuals = y_raw - fitted_values

    params = {
        "lookback": float(lookback),
        "channels": float(channels),
        "num_blocks": float(num_blocks),
        "kernel_size": float(kernel_size),
        "epochs": float(spec["epochs"]),
        "learning_rate": float(spec["learning_rate"]),
        "dropout": dropout,
    }

    return RecurrentSequenceFit(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        lookback=lookback,
        exogenous=exogenous,
        fittedvalues=pd.Series(fitted_values, index=pd.RangeIndex(len(fitted_values)), dtype=float),
        residuals=pd.Series(residuals, index=pd.RangeIndex(len(residuals)), dtype=float),
        params=params,
    )


def fit_nbeats(train_frame: pd.DataFrame) -> RecurrentSequenceFit:
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("N-BEATS requires torch and scikit-learn in the active environment.") from exc

    spec = get_active_spec()["nbeats"]
    exogenous = list(spec["exogenous"])
    lookback = int(spec["lookback"])
    x_raw, y_raw = recurrent_sequences(train_frame, lookback, exogenous)
    if len(x_raw) == 0:
        raise RuntimeError("N-BEATS requires more observations than the lookback window.")

    sample_count, _, feature_count = x_raw.shape
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_raw.reshape(sample_count * lookback, feature_count)).reshape(sample_count, lookback, feature_count)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)

    stack_width = int(spec["stack_width"])
    n_blocks = int(spec["n_blocks"])
    n_layers = int(spec["n_layers"])
    dropout = float(spec.get("dropout", 0.0))
    backcast_size = lookback * feature_count

    class NBeatsBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            input_size = backcast_size
            for _ in range(n_layers):
                layers.append(torch.nn.Linear(input_size, stack_width))
                layers.append(torch.nn.ReLU())
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(dropout))
                input_size = stack_width
            self.hidden = torch.nn.Sequential(*layers)
            self.backcast = torch.nn.Linear(stack_width, backcast_size)
            self.forecast = torch.nn.Linear(stack_width, 1)

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            theta = self.hidden(inputs)
            return self.backcast(theta), self.forecast(theta).squeeze(-1)

    class NBeatsRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([NBeatsBlock() for _ in range(n_blocks)])

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            residual = inputs.reshape(inputs.shape[0], -1)
            forecast = torch.zeros(inputs.shape[0], device=inputs.device)
            for block in self.blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast

    device = torch.device("cpu")
    model = NBeatsRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    model.train()
    for _ in range(int(spec["epochs"])):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        fitted_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1, 1)
    fitted_values = scaler_y.inverse_transform(fitted_scaled).reshape(-1)
    residuals = y_raw - fitted_values

    params = {
        "lookback": float(lookback),
        "stack_width": float(stack_width),
        "n_blocks": float(n_blocks),
        "n_layers": float(n_layers),
        "epochs": float(spec["epochs"]),
        "learning_rate": float(spec["learning_rate"]),
        "dropout": dropout,
    }

    return RecurrentSequenceFit(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        lookback=lookback,
        exogenous=exogenous,
        fittedvalues=pd.Series(fitted_values, index=pd.RangeIndex(len(fitted_values)), dtype=float),
        residuals=pd.Series(residuals, index=pd.RangeIndex(len(residuals)), dtype=float),
        params=params,
    )


def fit_transformer(train_frame: pd.DataFrame) -> RecurrentSequenceFit:
    try:
        import torch
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Transformer requires torch and scikit-learn in the active environment.") from exc

    spec = get_active_spec()["transformer"]
    exogenous = list(spec["exogenous"])
    lookback = int(spec["lookback"])
    x_raw, y_raw = recurrent_sequences(train_frame, lookback, exogenous)
    if len(x_raw) == 0:
        raise RuntimeError("Transformer requires more observations than the lookback window.")

    sample_count, _, feature_count = x_raw.shape
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_raw.reshape(sample_count * lookback, feature_count)).reshape(sample_count, lookback, feature_count)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)

    model_dim = int(spec["model_dim"])
    num_heads = int(spec["num_heads"])
    num_layers = int(spec["num_layers"])
    feedforward_dim = int(spec["feedforward_dim"])
    dropout = float(spec.get("dropout", 0.0))

    class TransformerRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(feature_count, model_dim)
            self.position_embedding = torch.nn.Parameter(torch.zeros(1, lookback, model_dim))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output = torch.nn.Linear(model_dim, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            embedded = self.input_projection(inputs) + self.position_embedding
            encoded = self.encoder(embedded)
            return self.output(encoded[:, -1, :]).squeeze(-1)

    device = torch.device("cpu")
    model = TransformerRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    model.train()
    for _ in range(int(spec["epochs"])):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        fitted_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1, 1)
    fitted_values = scaler_y.inverse_transform(fitted_scaled).reshape(-1)
    residuals = y_raw - fitted_values

    params = {
        "lookback": float(lookback),
        "model_dim": float(model_dim),
        "num_heads": float(num_heads),
        "num_layers": float(num_layers),
        "feedforward_dim": float(feedforward_dim),
        "epochs": float(spec["epochs"]),
        "learning_rate": float(spec["learning_rate"]),
        "dropout": dropout,
    }

    return RecurrentSequenceFit(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        lookback=lookback,
        exogenous=exogenous,
        fittedvalues=pd.Series(fitted_values, index=pd.RangeIndex(len(fitted_values)), dtype=float),
        residuals=pd.Series(residuals, index=pd.RangeIndex(len(residuals)), dtype=float),
        params=params,
    )


def recurrent_forecast_path(
    fitted: RecurrentSequenceFit,
    history_frame: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LSTM / GRU forecasting requires torch in the active environment.") from exc

    history = history_frame.copy().reset_index(drop=True)
    predictions: list[float] = []
    device = torch.device("cpu")
    fitted.model.eval()
    for _, future_row in future_exog.reset_index(drop=True).iterrows():
        window = recurrent_feature_matrix(history.tail(fitted.lookback), fitted.exogenous)
        window_scaled = fitted.scaler_x.transform(window)
        x_tensor = torch.tensor(window_scaled[np.newaxis, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_scaled = fitted.model(x_tensor).detach().cpu().numpy().reshape(-1, 1)
        prediction = float(fitted.scaler_y.inverse_transform(pred_scaled).reshape(-1)[0])
        predictions.append(prediction)
        next_row = {"date": future_row["date"], "hpi": prediction}
        for column in fitted.exogenous:
            next_row[column] = float(future_row[column])
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
    return np.asarray(predictions, dtype=float)


def regularized_linear_forecast_path(
    fitted: RegularizedLinearFit,
    history_frame: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> np.ndarray:
    history = history_frame.copy().reset_index(drop=True)
    predictions: list[float] = []
    feature_columns = fitted.feature_names
    for _, future_row in future_exog.reset_index(drop=True).iterrows():
        feature_values: dict[str, float] = {}
        for lag in fitted.target_lags:
            feature_values[f"hpi_lag_{lag}"] = float(history["hpi"].iloc[-lag])
        for column in fitted.exogenous:
            feature_values[column] = float(future_row[column])
        ordered = np.asarray([[feature_values[name] for name in feature_columns]], dtype=float)
        scaled = fitted.scaler.transform(ordered)
        prediction = float(fitted.estimator.predict(scaled)[0])
        predictions.append(prediction)
        next_row = {"date": future_row["date"], "hpi": prediction}
        for column in fitted.exogenous:
            next_row[column] = float(future_row[column])
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
    return np.asarray(predictions, dtype=float)


def tabular_ml_forecast_path(
    fitted: TabularMlFit,
    history_frame: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> np.ndarray:
    history = history_frame.copy().reset_index(drop=True)
    predictions: list[float] = []
    feature_columns = fitted.feature_names
    for _, future_row in future_exog.reset_index(drop=True).iterrows():
        feature_values: dict[str, float] = {}
        for lag in fitted.target_lags:
            feature_values[f"hpi_lag_{lag}"] = float(history["hpi"].iloc[-lag])
        for column in fitted.exogenous:
            feature_values[column] = float(future_row[column])
        ordered = np.asarray([[feature_values[name] for name in feature_columns]], dtype=float)
        if fitted.scaler is not None:
            ordered = fitted.scaler.transform(ordered)
        prediction = float(fitted.estimator.predict(ordered)[0])
        predictions.append(prediction)
        next_row = {"date": future_row["date"], "hpi": prediction}
        for column in fitted.exogenous:
            next_row[column] = float(future_row[column])
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
    return np.asarray(predictions, dtype=float)


def regularized_linear_interval(
    predictions: np.ndarray,
    residual_std: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_score = float(norm.ppf(1.0 - alpha / 2.0))
    horizons = np.sqrt(np.arange(1, len(predictions) + 1, dtype=float))
    width = z_score * residual_std * horizons
    return predictions - width, predictions + width


def fit_arimax(train_y: pd.Series, train_exog: pd.DataFrame | None):
    spec = get_active_spec()["arimax"]
    model = ARIMA(
        endog=train_y,
        exog=train_exog,
        order=tuple(spec["order"]),
        trend=spec["trend"],
        enforce_stationarity=spec["enforce_stationarity"],
        enforce_invertibility=spec["enforce_invertibility"],
    )
    return model.fit(method_kwargs={"maxiter": spec["maxiter"]})


def fit_arimax_garch(train_y: pd.Series, train_exog: pd.DataFrame | None):
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ARIMAX-GARCH requires the 'arch' package in the active environment.") from exc

    spec = get_active_spec()["arimax_garch"]
    mean_model = ARIMA(
        endog=train_y,
        exog=train_exog,
        order=tuple(spec["order"]),
        trend=spec["trend"],
        enforce_stationarity=spec["enforce_stationarity"],
        enforce_invertibility=spec["enforce_invertibility"],
    )
    mean_result = mean_model.fit(method_kwargs={"maxiter": spec["maxiter"]})
    residuals = pd.Series(np.asarray(mean_result.resid, dtype=float), index=train_y.index).dropna()
    residual_scale = float(spec.get("residual_scale", 100.0))
    scaled_residuals = residuals * residual_scale

    garch_model = arch_model(
        scaled_residuals,
        mean="Zero",
        vol=spec["volatility"],
        p=int(spec["p"]),
        o=int(spec.get("o", 0)),
        q=int(spec["q"]),
        dist=spec["distribution"],
        rescale=False,
    )
    volatility_result = garch_model.fit(disp="off", show_warning=False)
    return ArimaxGarchFit(mean_result, volatility_result, residual_scale, spec["distribution"])


def arimax_garch_nu(fitted: ArimaxGarchFit) -> float:
    nu = float(model_param_lookup(fitted.volatility_result).get("nu", 8.0))
    return max(nu, 2.1)


def arimax_garch_forecast_sigma(fitted: ArimaxGarchFit, steps: int) -> np.ndarray:
    variance = fitted.volatility_result.forecast(horizon=steps, reindex=False).variance
    sigma = np.sqrt(np.maximum(variance.iloc[-1].to_numpy(dtype=float), 1e-12))
    return sigma / fitted.residual_scale


def arimax_garch_interval_delta(fitted: ArimaxGarchFit, sigma: np.ndarray, alpha: float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    if fitted.distribution == "t":
        nu = arimax_garch_nu(fitted)
        scale = sigma * np.sqrt((nu - 2.0) / nu)
        return student_t.ppf(1.0 - alpha / 2.0, df=nu) * scale
    return norm.ppf(1.0 - alpha / 2.0) * sigma


def arimax_garch_logpdf(fitted: ArimaxGarchFit, errors: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    errors = np.asarray(errors, dtype=float)
    if fitted.distribution == "t":
        nu = arimax_garch_nu(fitted)
        scale = np.maximum(sigma * np.sqrt((nu - 2.0) / nu), 1e-6)
        return student_t.logpdf(errors, df=nu, loc=0.0, scale=scale)
    return norm.logpdf(errors, loc=0.0, scale=sigma)


def arimax_garch_standardized_residuals(fitted: ArimaxGarchFit) -> pd.Series:
    values = pd.Series(np.asarray(fitted.volatility_result.std_resid, dtype=float))
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values


def standardize_series(values: pd.Series) -> tuple[pd.Series, float, float]:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 0.0 or not np.isfinite(std):
        std = 1.0
    return ((values - mean) / std).astype(float), mean, std


def standardize_frame(frame: pd.DataFrame | None) -> tuple[pd.DataFrame | None, dict[str, float], dict[str, float]]:
    if frame is None:
        return None, {}, {}

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    scaled = frame.copy()
    for column in scaled.columns:
        mean = float(scaled[column].mean())
        std = float(scaled[column].std(ddof=0))
        if std <= 0.0 or not np.isfinite(std):
            std = 1.0
        means[column] = mean
        stds[column] = std
        scaled[column] = ((scaled[column] - mean) / std).astype(float)
    return scaled, means, stds


def inverse_scaled_values(values: Any, mean: float, std: float) -> np.ndarray:
    return np.asarray(values, dtype=float) * std + mean


def state_space_attempt_grid() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_lbfgs", "method": "lbfgs", "maxiter": 50, "scale_inputs": False},
        {"name": "lbfgs_200", "method": "lbfgs", "maxiter": 200, "scale_inputs": False},
        {"name": "powell_400", "method": "powell", "maxiter": 400, "scale_inputs": False},
        {"name": "scaled_lbfgs_200", "method": "lbfgs", "maxiter": 200, "scale_inputs": True},
        {"name": "scaled_powell_400", "method": "powell", "maxiter": 400, "scale_inputs": True},
    ]


def dynamic_factor_attempt_grid() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_lbfgs", "method": "lbfgs", "maxiter": 200, "scale_inputs": False, "enforce_stationarity": True},
        {"name": "relaxed_lbfgs", "method": "lbfgs", "maxiter": 200, "scale_inputs": False, "enforce_stationarity": False},
        {"name": "relaxed_powell", "method": "powell", "maxiter": 400, "scale_inputs": False, "enforce_stationarity": False},
        {"name": "scaled_relaxed_lbfgs", "method": "lbfgs", "maxiter": 200, "scale_inputs": True, "enforce_stationarity": False},
    ]


def attach_state_space_fit_metadata(
    fitted: Any,
    attempt: dict[str, Any],
    attempt_history: list[dict[str, Any]],
    target_mean: float,
    target_std: float,
    exog_means: dict[str, float],
    exog_stds: dict[str, float],
    recovered: bool,
) -> Any:
    fitted._fit_attempts = copy.deepcopy(attempt_history)
    fitted._fit_strategy = attempt["name"]
    fitted._fit_scaled = bool(attempt["scale_inputs"])
    fitted._fit_recovered = bool(recovered)
    fitted._target_mean = float(target_mean)
    fitted._target_std = float(target_std)
    fitted._exog_means = exog_means
    fitted._exog_stds = exog_stds
    return fitted


def fit_ardl(train_frame: pd.DataFrame):
    spec = get_active_spec()["ardl"]
    exog_columns = spec["exogenous"]
    ar_lags = int(spec["ar_lags"])
    distributed_lags = int(spec["distributed_lags"])
    trend = spec["trend"]
    exog = build_exog(train_frame, exog_columns)
    model = ARDL(
        endog=train_frame["hpi"].astype(float),
        lags=ar_lags,
        exog=exog,
        order=distributed_lags if exog is not None else 0,
        trend=trend,
    )
    return model.fit()


def fit_uecm(train_frame: pd.DataFrame):
    spec = get_active_spec()["uecm"]
    exog_columns = spec["exogenous"]
    ar_lags = int(spec["ar_lags"])
    distributed_lags = int(spec["distributed_lags"])
    trend = spec["trend"]
    exog = build_exog(train_frame, exog_columns)
    model = UECM(
        endog=train_frame["hpi"].astype(float),
        lags=ar_lags,
        exog=exog,
        order=distributed_lags if exog is not None else 0,
        trend=trend,
    )
    return model.fit()


def fit_ets(train_frame: pd.DataFrame):
    spec = get_active_spec()["ets"]
    model = ETSModel(
        endog=train_frame["hpi"].astype(float),
        error=spec["error"],
        trend=spec["trend"],
        damped_trend=bool(spec["damped_trend"]),
        seasonal=None,
    )
    return model.fit(disp=False)


def fit_dols_fmols(train_frame: pd.DataFrame) -> CointegrationRegressionFit:
    spec = get_active_spec()["dols_fmols"]
    exog_columns = list(spec["exogenous"])
    if not exog_columns:
        raise RuntimeError("DOLS / FMOLS requires at least one exogenous variable.")

    y = train_frame["hpi"].astype(float)
    x = train_frame[exog_columns].astype(float)
    method = str(spec["method"])
    trend = str(spec["trend"])

    if method == "dols":
        leads = int(spec["leads"])
        lags = int(spec["lags"])
        estimator = DynamicOLS(y, x, trend=trend, leads=leads, lags=lags).fit(cov_type="robust")
    elif method == "fmols":
        leads = 0
        lags = 0
        estimator = FullyModifiedOLS(y, x, trend=trend).fit()
    else:
        raise RuntimeError(f"Unsupported cointegration regression method: {method}")

    params = pd.Series(estimator.params, dtype=float)
    fitted_index = estimator.resid.index
    fitted_y = train_frame.loc[fitted_index, "hpi"].astype(float)
    fittedvalues = fitted_y - pd.Series(estimator.resid, index=fitted_index, dtype=float)
    residuals = pd.Series(estimator.resid, index=fitted_index, dtype=float)
    return CointegrationRegressionFit(
        estimator=estimator,
        method=method,
        exogenous=exog_columns,
        trend=trend,
        fittedvalues=fittedvalues,
        residuals=residuals,
        params=params,
        leads=leads,
        lags=lags,
    )


def fit_markov_switching(train_frame: pd.DataFrame) -> MarkovSwitchingFit:
    spec = get_active_spec()["markov_switching"]
    exog_columns = list(spec["exogenous"])
    exog = build_exog(train_frame, exog_columns)
    attempt_grid = [
        {"name": "bfgs_basic", "method": "bfgs", "maxiter": 100, "search_reps": 0},
        {"name": "bfgs_search", "method": "bfgs", "maxiter": 200, "search_reps": 10},
        {"name": "lbfgs_search", "method": "lbfgs", "maxiter": 200, "search_reps": 10},
    ]
    last_error: Exception | None = None
    for attempt in attempt_grid:
        try:
            model = MarkovRegression(
                endog=train_frame["hpi"].astype(float),
                k_regimes=2,
                trend="c",
                exog=exog,
                order=0,
                switching_trend=True,
                switching_exog=False,
                switching_variance=True,
            )
            estimator = model.fit(
                disp=False,
                method=attempt["method"],
                maxiter=attempt["maxiter"],
                search_reps=attempt["search_reps"],
            )
            fittedvalues = pd.Series(np.asarray(estimator.fittedvalues, dtype=float), index=train_frame.index)
            residuals = pd.Series(np.asarray(estimator.resid, dtype=float), index=train_frame.index)
            transition_matrix = np.asarray(estimator.regime_transition, dtype=float)[:, :, -1]
            last_probabilities = np.asarray(estimator.smoothed_marginal_probabilities.iloc[-1], dtype=float)
            return MarkovSwitchingFit(
                estimator=estimator,
                exogenous=exog_columns,
                params=pd.Series(estimator.params, dtype=float),
                fittedvalues=fittedvalues,
                residuals=residuals,
                transition_matrix=transition_matrix,
                last_probabilities=last_probabilities,
                fit_strategy=attempt["name"],
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Markov-Switching fit failed after retries: {last_error}")


def markov_switching_forecast_path(
    fitted: MarkovSwitchingFit,
    future_exog: pd.DataFrame,
) -> np.ndarray:
    probabilities = fitted.last_probabilities.astype(float)
    const_by_regime = np.asarray(
        [
            float(fitted.params.get("const[0]", 0.0)),
            float(fitted.params.get("const[1]", 0.0)),
        ],
        dtype=float,
    )
    shared_exog_coef: dict[str, float] = {}
    for idx, variable in enumerate(fitted.exogenous, start=1):
        shared_exog_coef[variable] = float(fitted.params.get(f"x{idx}[1]", 0.0))

    forecasts: list[float] = []
    for _, row in future_exog.iterrows():
        probabilities = fitted.transition_matrix @ probabilities
        regime_means = const_by_regime.copy()
        for variable in fitted.exogenous:
            regime_means += shared_exog_coef[variable] * float(row[variable])
        forecasts.append(float(probabilities @ regime_means))
    return np.asarray(forecasts, dtype=float)


def cointegration_regression_predict(
    fitted: CointegrationRegressionFit,
    exog_frame: pd.DataFrame,
    index: pd.Index,
) -> pd.Series:
    frame = exog_frame[fitted.exogenous].astype(float).copy()
    values = pd.Series(0.0, index=index, dtype=float)
    for variable in fitted.exogenous:
        if variable in fitted.params.index:
            values = values + float(fitted.params[variable]) * frame[variable].to_numpy(dtype=float)
    if "const" in fitted.params.index:
        values = values + float(fitted.params["const"])
    if "trend" in fitted.params.index:
        trend_start = len(index)
        if len(index) and isinstance(index[0], (int, np.integer)):
            trend_values = np.asarray(index, dtype=float) + 1.0
        else:
            trend_values = np.arange(trend_start, trend_start + len(index), dtype=float)
        values = values + float(fitted.params["trend"]) * trend_values
    return pd.Series(values, index=index, dtype=float)


def uecm_forecast_path(
    fitted: Any,
    history_y: pd.Series,
    history_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> np.ndarray:
    params = pd.Series(fitted.params, dtype=float)
    state_y = history_y.astype(float).tolist()
    exog_columns = list(history_exog.columns)
    state_x = {column: history_exog[column].astype(float).tolist() for column in exog_columns}
    forecasts: list[float] = []

    for _, future_row in future_exog.iterrows():
        current_x = {column: float(future_row[column]) for column in exog_columns}
        delta_y = 0.0
        for name, coefficient in params.items():
            if name == "const":
                delta_y += float(coefficient)
            elif name == "trend":
                delta_y += float(coefficient) * float(len(state_y) + 1)
            elif name in {"y.L1", "hpi.L1"}:
                delta_y += float(coefficient) * float(state_y[-1])
            elif name.startswith("D.y.L") or name.startswith("D.hpi.L"):
                lag = int(name.rsplit("L", 1)[1])
                dy_history = np.diff(np.asarray(state_y, dtype=float))
                if len(dy_history) >= lag:
                    delta_y += float(coefficient) * float(dy_history[-lag])
            elif name.endswith(".L1") and not name.startswith("D."):
                variable = name[: -len(".L1")]
                if variable in state_x:
                    delta_y += float(coefficient) * float(state_x[variable][-1])
            elif name.startswith("D.") and ".L" in name:
                variable, lag_text = name[2:].rsplit(".L", 1)
                lag = int(lag_text)
                if variable not in state_x:
                    continue
                if lag == 0:
                    delta_y += float(coefficient) * float(current_x[variable] - state_x[variable][-1])
                else:
                    dx_history = np.diff(np.asarray(state_x[variable], dtype=float))
                    if len(dx_history) >= lag:
                        delta_y += float(coefficient) * float(dx_history[-lag])

        next_y = float(state_y[-1] + delta_y)
        forecasts.append(next_y)
        state_y.append(next_y)
        for column in exog_columns:
            state_x[column].append(current_x[column])

    return np.asarray(forecasts, dtype=float)


def attach_dynamic_factor_fit_metadata(
    fitted: Any,
    attempt: dict[str, Any],
    attempt_history: list[dict[str, Any]],
    target_means: dict[str, float],
    target_stds: dict[str, float],
    recovered: bool,
) -> Any:
    fitted._fit_attempts = copy.deepcopy(attempt_history)
    fitted._fit_strategy = attempt["name"]
    fitted._fit_scaled = bool(attempt["scale_inputs"])
    fitted._fit_recovered = bool(recovered)
    fitted._target_means = target_means
    fitted._target_stds = target_stds
    return fitted


def state_space_future_exog(fitted: Any, future_exog: pd.DataFrame | None) -> pd.DataFrame | None:
    if future_exog is None:
        return None
    if not getattr(fitted, "_fit_scaled", False):
        return future_exog.astype(float)

    transformed = future_exog.copy().astype(float)
    means = getattr(fitted, "_exog_means", {})
    stds = getattr(fitted, "_exog_stds", {})
    for column in transformed.columns:
        mean = float(means.get(column, 0.0))
        std = float(stds.get(column, 1.0))
        if std <= 0.0 or not np.isfinite(std):
            std = 1.0
        transformed[column] = (transformed[column] - mean) / std
    return transformed


def state_space_fittedvalues_series(fitted: Any, index: pd.Index) -> pd.Series:
    values = np.asarray(fitted.fittedvalues, dtype=float)
    if getattr(fitted, "_fit_scaled", False):
        values = inverse_scaled_values(values, fitted._target_mean, fitted._target_std)
    return pd.Series(values, index=index)


def state_space_residuals_series(fitted: Any, actual_y: pd.Series) -> pd.Series:
    predicted = state_space_fittedvalues_series(fitted, actual_y.index)
    return actual_y.astype(float) - predicted


def state_space_forecast_components(
    fitted: Any,
    steps: int,
    future_exog: pd.DataFrame | None,
    alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    exog_fc = state_space_future_exog(fitted, future_exog)
    forecast = fitted.get_forecast(steps=steps, exog=exog_fc)
    predicted = np.asarray(forecast.predicted_mean, dtype=float)
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    if alpha is not None:
        conf = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
        lower = conf[:, 0]
        upper = conf[:, 1]
    if getattr(fitted, "_fit_scaled", False):
        predicted = inverse_scaled_values(predicted, fitted._target_mean, fitted._target_std)
        if lower is not None and upper is not None:
            lower = inverse_scaled_values(lower, fitted._target_mean, fitted._target_std)
            upper = inverse_scaled_values(upper, fitted._target_mean, fitted._target_std)
    return predicted, lower, upper


def dynamic_factor_fittedvalues_frame(fitted: Any, columns: list[str], index: pd.Index) -> pd.DataFrame:
    values = np.asarray(fitted.fittedvalues, dtype=float)
    frame = pd.DataFrame(values, columns=columns, index=index)
    if getattr(fitted, "_fit_scaled", False):
        means = getattr(fitted, "_target_means", {})
        stds = getattr(fitted, "_target_stds", {})
        for column in columns:
            frame[column] = frame[column] * float(stds.get(column, 1.0)) + float(means.get(column, 0.0))
    return frame


def dynamic_factor_forecast_components(
    fitted: Any,
    columns: list[str],
    steps: int,
    alpha: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    forecast = fitted.get_forecast(steps=steps)
    predicted = pd.DataFrame(np.asarray(forecast.predicted_mean, dtype=float), columns=columns)
    lower: pd.DataFrame | None = None
    upper: pd.DataFrame | None = None
    if alpha is not None:
        conf = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
        lower = pd.DataFrame({column: conf[:, 2 * idx] for idx, column in enumerate(columns)})
        upper = pd.DataFrame({column: conf[:, 2 * idx + 1] for idx, column in enumerate(columns)})
    if getattr(fitted, "_fit_scaled", False):
        means = getattr(fitted, "_target_means", {})
        stds = getattr(fitted, "_target_stds", {})
        for column in columns:
            std = float(stds.get(column, 1.0))
            mean = float(means.get(column, 0.0))
            predicted[column] = predicted[column] * std + mean
            if lower is not None and upper is not None:
                lower[column] = lower[column] * std + mean
                upper[column] = upper[column] * std + mean
    return predicted, lower, upper


def fit_state_space(train_y: pd.Series, train_exog: pd.DataFrame | None):
    spec = get_active_spec()["state_space"]
    attempt_history: list[dict[str, Any]] = []
    last_exception: Exception | None = None

    for index, attempt in enumerate(state_space_attempt_grid(), start=1):
        scale_inputs = bool(attempt["scale_inputs"])
        if scale_inputs:
            y_fit, target_mean, target_std = standardize_series(train_y.astype(float))
            exog_fit, exog_means, exog_stds = standardize_frame(train_exog.astype(float) if train_exog is not None else None)
        else:
            y_fit = train_y.astype(float)
            exog_fit = train_exog.astype(float) if train_exog is not None else None
            target_mean = 0.0
            target_std = 1.0
            exog_means = {}
            exog_stds = {}

        model = UnobservedComponents(
            endog=y_fit,
            level=spec["level"],
            cycle=spec["cycle"],
            stochastic_cycle=spec["stochastic_cycle"],
            damped_cycle=spec["damped_cycle"],
            exog=exog_fit,
        )

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = model.fit(
                    disp=False,
                    method=attempt["method"],
                    maxiter=attempt["maxiter"],
                )
        except Exception as exc:
            last_exception = exc
            attempt_history.append(
                {
                    "attempt": index,
                    "name": attempt["name"],
                    "method": attempt["method"],
                    "maxiter": attempt["maxiter"],
                    "scale_inputs": scale_inputs,
                    "status": "exception",
                    "exception": str(exc),
                }
            )
            continue

        warning_messages = [str(item.message) for item in caught]
        convergence_warned = any(
            issubclass(item.category, ConvergenceWarning) or "converg" in str(item.message).lower()
            for item in caught
        )
        runtime_warned = any(issubclass(item.category, RuntimeWarning) for item in caught)
        mle_retvals = getattr(fitted, "mle_retvals", {}) or {}
        optimizer_converged = bool(mle_retvals.get("converged", not convergence_warned))
        converged_cleanly = optimizer_converged and not convergence_warned and not runtime_warned
        attempt_history.append(
            {
                "attempt": index,
                "name": attempt["name"],
                "method": attempt["method"],
                "maxiter": attempt["maxiter"],
                "scale_inputs": scale_inputs,
                "status": "ok" if converged_cleanly else "warning",
                "optimizer_converged": optimizer_converged,
                "warnings": warning_messages,
            }
        )
        if converged_cleanly:
            return attach_state_space_fit_metadata(
                fitted,
                attempt,
                attempt_history,
                target_mean,
                target_std,
                exog_means,
                exog_stds,
                recovered=index > 1,
            )

    if last_exception is not None:
        raise RuntimeError(f"State-Space fit failed after recovery attempts: {last_exception}") from last_exception
    raise RuntimeError(f"State-Space fit did not converge after recovery attempts: {json.dumps(attempt_history)}")


def fit_var(train_frame: pd.DataFrame):
    spec = get_active_spec()["var"]
    endog_columns = spec["endogenous"]
    model = VAR(train_frame[endog_columns].astype(float))
    return model.fit(maxlags=spec["lags"], trend=spec["trend"])


def bvar_design_matrix(frame: pd.DataFrame, endog_columns: list[str], lags: int, trend: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    values = frame[endog_columns].astype(float).to_numpy()
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for idx in range(lags, len(frame)):
        lag_blocks = [values[idx - lag] for lag in range(1, lags + 1)]
        design = np.concatenate(lag_blocks)
        if trend == "c":
            design = np.concatenate(([1.0], design))
        rows.append(design)
        targets.append(values[idx])
    if not rows:
        raise RuntimeError("Not enough rows to estimate BVAR with the selected lag order.")
    return np.vstack(rows), np.vstack(targets), frame.index[lags:]


def threshold_design_components(
    frame: pd.DataFrame,
    endog_columns: list[str],
    lags: int,
    trend: str,
    threshold_variable: str,
    delay: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index, np.ndarray]:
    values = frame[endog_columns].astype(float).to_numpy()
    threshold_idx = endog_columns.index(threshold_variable)
    start = max(lags, delay)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    threshold_values: list[float] = []
    for idx in range(start, len(frame)):
        lag_blocks = [values[idx - lag] for lag in range(1, lags + 1)]
        design = np.concatenate(lag_blocks)
        if trend == "c":
            design = np.concatenate(([1.0], design))
        rows.append(design)
        targets.append(values[idx])
        threshold_values.append(float(values[idx - delay, threshold_idx]))
    if not rows:
        raise RuntimeError("Not enough rows to estimate Threshold VAR / SETAR with the selected lag and delay.")
    return np.vstack(rows), np.vstack(targets), frame.index[start:], np.asarray(threshold_values, dtype=float)


def ridge_multivariate_solution(x_matrix: np.ndarray, y_matrix: np.ndarray, ridge_alpha: float, penalize_intercept: bool) -> np.ndarray:
    penalty = np.eye(x_matrix.shape[1], dtype=float) * float(ridge_alpha)
    if not penalize_intercept and x_matrix.shape[1] > 0:
        penalty[0, 0] = 0.0
    return np.linalg.solve(x_matrix.T @ x_matrix + penalty, x_matrix.T @ y_matrix)


def fit_bvar(train_frame: pd.DataFrame):
    spec = get_active_spec()["bvar"]
    endog_columns = spec["endogenous"]
    lags = int(spec["lags"])
    trend = spec["trend"]
    tightness = float(spec["tightness"])
    prior_type = spec["prior_type"]
    lag_decay = float(spec.get("lag_decay", 1.0))

    x_matrix, y_matrix, fitted_index = bvar_design_matrix(train_frame, endog_columns, lags, trend)
    n_obs, n_features = x_matrix.shape
    n_endog = len(endog_columns)

    prior_mean = np.zeros((n_features, n_endog), dtype=float)
    prior_precision = np.zeros(n_features, dtype=float)
    offset = 1 if trend == "c" else 0
    if trend == "c":
        prior_precision[0] = 0.01

    for lag in range(1, lags + 1):
        weight = 1.0 / ((tightness * (lag ** lag_decay)) ** 2)
        for var_idx in range(n_endog):
            position = offset + (lag - 1) * n_endog + var_idx
            prior_precision[position] = weight
            if lag == 1:
                prior_mean[position, var_idx] = 1.0

    precision_matrix = np.diag(prior_precision)
    coefficients = np.zeros((n_features, n_endog), dtype=float)
    xtx = x_matrix.T @ x_matrix
    for eq_idx in range(n_endog):
        posterior_precision = xtx + precision_matrix
        posterior_mean = x_matrix.T @ y_matrix[:, eq_idx] + precision_matrix @ prior_mean[:, eq_idx]
        coefficients[:, eq_idx] = np.linalg.solve(posterior_precision, posterior_mean)

    fitted_values = x_matrix @ coefficients
    residuals = y_matrix - fitted_values
    sigma_u = (residuals.T @ residuals) / max(n_obs - n_features, 1)
    sigma_u += np.eye(n_endog) * 1e-6

    if trend == "c":
        intercept = coefficients[0, :]
        lag_coefficients = coefficients[1:, :].reshape(lags, n_endog, n_endog)
    else:
        intercept = np.zeros(n_endog, dtype=float)
        lag_coefficients = coefficients.reshape(lags, n_endog, n_endog)

    fitted_frame = pd.DataFrame(fitted_values, columns=endog_columns, index=fitted_index)
    residual_frame = pd.DataFrame(residuals, columns=endog_columns, index=fitted_index)
    return BVarFit(
        coefficients=lag_coefficients,
        intercept=intercept,
        sigma_u=sigma_u,
        lags=lags,
        trend=trend,
        endogenous=endog_columns,
        fittedvalues=fitted_frame,
        residuals=residual_frame,
        train_frame=train_frame[endog_columns].copy(),
        prior_type=prior_type,
        tightness=tightness,
    )


def fit_threshold_var(train_frame: pd.DataFrame):
    spec = get_active_spec()["threshold_var"]
    endog_columns = spec["endogenous"]
    lags = int(spec["lags"])
    trend = spec["trend"]
    threshold_variable = spec["threshold_variable"]
    delay = int(spec["delay"])
    threshold_quantile = float(spec["threshold_quantile"])
    ridge_alpha = float(spec["ridge_alpha"])

    x_matrix, y_matrix, fitted_index, threshold_history = threshold_design_components(
        train_frame,
        endog_columns,
        lags,
        trend,
        threshold_variable,
        delay,
    )
    threshold_value = float(np.quantile(threshold_history, threshold_quantile))
    low_mask = threshold_history <= threshold_value
    high_mask = ~low_mask
    n_features = x_matrix.shape[1]
    min_regime_obs = max(n_features + 2, 12)
    if int(low_mask.sum()) < min_regime_obs or int(high_mask.sum()) < min_regime_obs:
        raise RuntimeError(
            f"Threshold split too imbalanced for {threshold_variable} q={threshold_quantile}: "
            f"low={int(low_mask.sum())}, high={int(high_mask.sum())}, need at least {min_regime_obs}."
        )

    low_beta = ridge_multivariate_solution(x_matrix[low_mask], y_matrix[low_mask], ridge_alpha, penalize_intercept=(trend != "c"))
    high_beta = ridge_multivariate_solution(x_matrix[high_mask], y_matrix[high_mask], ridge_alpha, penalize_intercept=(trend != "c"))
    fitted_values = np.empty_like(y_matrix)
    fitted_values[low_mask] = x_matrix[low_mask] @ low_beta
    fitted_values[high_mask] = x_matrix[high_mask] @ high_beta
    residuals = y_matrix - fitted_values
    sigma_u = (residuals.T @ residuals) / max(len(y_matrix) - n_features, 1)
    sigma_u += np.eye(len(endog_columns)) * 1e-6

    if trend == "c":
        low_intercept = low_beta[0, :]
        high_intercept = high_beta[0, :]
        low_coefficients = low_beta[1:, :].reshape(lags, len(endog_columns), len(endog_columns))
        high_coefficients = high_beta[1:, :].reshape(lags, len(endog_columns), len(endog_columns))
    else:
        low_intercept = np.zeros(len(endog_columns), dtype=float)
        high_intercept = np.zeros(len(endog_columns), dtype=float)
        low_coefficients = low_beta.reshape(lags, len(endog_columns), len(endog_columns))
        high_coefficients = high_beta.reshape(lags, len(endog_columns), len(endog_columns))

    fitted_frame = pd.DataFrame(fitted_values, columns=endog_columns, index=fitted_index)
    residual_frame = pd.DataFrame(residuals, columns=endog_columns, index=fitted_index)
    return ThresholdVarFit(
        low_coefficients=low_coefficients,
        high_coefficients=high_coefficients,
        low_intercept=low_intercept,
        high_intercept=high_intercept,
        sigma_u=sigma_u,
        lags=lags,
        trend=trend,
        endogenous=endog_columns,
        threshold_variable=threshold_variable,
        threshold_value=threshold_value,
        delay=delay,
        fittedvalues=fitted_frame,
        residuals=residual_frame,
        train_frame=train_frame[endog_columns].copy(),
        threshold_quantile=threshold_quantile,
        regime_counts={"low": int(low_mask.sum()), "high": int(high_mask.sum())},
    )


def bvar_forecast(fitted: BVarFit, history: np.ndarray, steps: int) -> np.ndarray:
    state = [row.astype(float).copy() for row in history[-fitted.lags :]]
    forecasts: list[np.ndarray] = []
    for _ in range(steps):
        next_value = fitted.intercept.copy()
        for lag in range(fitted.lags):
            next_value = next_value + fitted.coefficients[lag] @ state[-(lag + 1)]
        forecasts.append(next_value.copy())
        state.append(next_value.copy())
    return np.vstack(forecasts)


def threshold_var_next_value(fitted: ThresholdVarFit, state: list[np.ndarray]) -> np.ndarray:
    threshold_idx = fitted.endogenous.index(fitted.threshold_variable)
    trigger_value = float(state[-fitted.delay][threshold_idx])
    use_low_regime = trigger_value <= fitted.threshold_value
    intercept = fitted.low_intercept if use_low_regime else fitted.high_intercept
    coefficients = fitted.low_coefficients if use_low_regime else fitted.high_coefficients
    next_value = intercept.copy()
    for lag in range(fitted.lags):
        next_value = next_value + coefficients[lag] @ state[-(lag + 1)]
    return next_value


def threshold_var_forecast(fitted: ThresholdVarFit, history: np.ndarray, steps: int) -> np.ndarray:
    state = [row.astype(float).copy() for row in history[-fitted.lags :]]
    forecasts: list[np.ndarray] = []
    for _ in range(steps):
        next_value = threshold_var_next_value(fitted, state)
        forecasts.append(next_value.copy())
        state.append(next_value.copy())
    return np.vstack(forecasts)


def threshold_var_forecast_interval(
    fitted: ThresholdVarFit,
    history: np.ndarray,
    steps: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast = threshold_var_forecast(fitted, history, steps)
    sigma_hpi = float(np.sqrt(max(fitted.sigma_u[fitted.endogenous.index("hpi"), fitted.endogenous.index("hpi")], 1e-6)))
    scales = sigma_hpi * np.sqrt(np.arange(1, steps + 1, dtype=float))
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    lower = forecast.copy()
    upper = forecast.copy()
    hpi_idx = fitted.endogenous.index("hpi")
    lower[:, hpi_idx] = forecast[:, hpi_idx] - z_value * scales
    upper[:, hpi_idx] = forecast[:, hpi_idx] + z_value * scales
    return forecast, lower, upper


def bvar_forecast_interval(fitted: BVarFit, history: np.ndarray, steps: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast = bvar_forecast(fitted, history, steps)
    sigma_hpi = float(np.sqrt(max(fitted.sigma_u[fitted.endogenous.index("hpi"), fitted.endogenous.index("hpi")], 1e-6)))
    scales = sigma_hpi * np.sqrt(np.arange(1, steps + 1, dtype=float))
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    lower = forecast.copy()
    upper = forecast.copy()
    hpi_idx = fitted.endogenous.index("hpi")
    lower[:, hpi_idx] = forecast[:, hpi_idx] - z_value * scales
    upper[:, hpi_idx] = forecast[:, hpi_idx] + z_value * scales
    return forecast, lower, upper


def fit_dynamic_factor(train_frame: pd.DataFrame):
    spec = get_active_spec()["dynamic_factor"]
    endog_columns = spec["endogenous"]
    attempt_history: list[dict[str, Any]] = []
    last_exception: Exception | None = None

    for index, attempt in enumerate(dynamic_factor_attempt_grid(), start=1):
        scale_inputs = bool(attempt["scale_inputs"])
        if scale_inputs:
            endog_fit, target_means, target_stds = standardize_frame(train_frame[endog_columns].astype(float))
        else:
            endog_fit = train_frame[endog_columns].astype(float)
            target_means = {}
            target_stds = {}

        model = DynamicFactor(
            endog=endog_fit,
            k_factors=spec["k_factors"],
            factor_order=spec["factor_order"],
            error_order=spec["error_order"],
            error_var=spec["error_var"],
            error_cov_type=spec["error_cov_type"],
            enforce_stationarity=attempt["enforce_stationarity"],
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = model.fit(
                    disp=False,
                    method=attempt["method"],
                    maxiter=attempt["maxiter"],
                )
        except Exception as exc:
            last_exception = exc
            attempt_history.append(
                {
                    "attempt": index,
                    "name": attempt["name"],
                    "method": attempt["method"],
                    "maxiter": attempt["maxiter"],
                    "scale_inputs": scale_inputs,
                    "enforce_stationarity": attempt["enforce_stationarity"],
                    "status": "exception",
                    "exception": str(exc),
                }
            )
            continue

        warning_messages = [str(item.message) for item in caught]
        convergence_warned = any(
            issubclass(item.category, ConvergenceWarning) or "converg" in str(item.message).lower()
            for item in caught
        )
        runtime_warned = any(issubclass(item.category, RuntimeWarning) for item in caught)
        mle_retvals = getattr(fitted, "mle_retvals", {}) or {}
        optimizer_converged = bool(mle_retvals.get("converged", not convergence_warned))
        converged_cleanly = optimizer_converged and not convergence_warned and not runtime_warned
        attempt_history.append(
            {
                "attempt": index,
                "name": attempt["name"],
                "method": attempt["method"],
                "maxiter": attempt["maxiter"],
                "scale_inputs": scale_inputs,
                "enforce_stationarity": attempt["enforce_stationarity"],
                "status": "ok" if converged_cleanly else "warning",
                "optimizer_converged": optimizer_converged,
                "warnings": warning_messages,
            }
        )
        if converged_cleanly:
            return attach_dynamic_factor_fit_metadata(
                fitted,
                attempt,
                attempt_history,
                target_means,
                target_stds,
                recovered=index > 1,
            )

    if last_exception is not None:
        raise RuntimeError(f"Dynamic Factor fit failed after recovery attempts: {last_exception}") from last_exception
    raise RuntimeError(f"Dynamic Factor fit did not converge after recovery attempts: {json.dumps(attempt_history)}")


def resolve_vecm_rank(train_frame: pd.DataFrame, endog_columns: list[str], deterministic: str, k_ar_diff: int) -> int:
    det_order_map = {
        "n": -1,
        "ci": 0,
        "co": 0,
        "li": 1,
        "lo": 1,
    }
    endog = train_frame[endog_columns].astype(float).to_numpy()
    rank_result = select_coint_rank(
        endog=endog,
        det_order=det_order_map.get(deterministic, 0),
        k_ar_diff=k_ar_diff,
        method="trace",
        signif=0.05,
    )
    max_rank = max(1, len(endog_columns) - 1)
    return max(1, min(int(rank_result.rank), max_rank))


def fit_vecm(train_frame: pd.DataFrame):
    spec = get_active_spec()["vecm"]
    endog_columns = spec["endogenous"]
    k_ar_diff = int(spec["k_ar_diff"])
    deterministic = spec["deterministic"]
    coint_rank = resolve_vecm_rank(train_frame, endog_columns, deterministic, k_ar_diff)
    model = VECM(
        endog=train_frame[endog_columns].astype(float),
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic,
    )
    fitted = model.fit()
    fitted._resolved_coint_rank = coint_rank
    fitted._resolved_endog_columns = endog_columns
    return fitted


def walk_forward_validation(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["arimax"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_arimax(train["hpi"], build_exog(train, exog_columns))
        forecast = fitted.get_forecast(steps=len(future), exog=build_exog(future, exog_columns))
        predicted = pd.Series(np.asarray(forecast.predicted_mean), index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_ardl(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["ardl"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_ardl(train)
        predicted = pd.Series(
            uecm_forecast_path(
                fitted,
                train["hpi"],
                train[exog_columns],
                future[exog_columns],
            ),
            index=future.index,
        )
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_uecm(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["uecm"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_uecm(train)
        predicted = pd.Series(
            uecm_forecast_path(
                fitted,
                train["hpi"],
                train[exog_columns],
                future[exog_columns],
            ),
            index=future.index,
        )
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_ets(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_ets(train)
        predicted = pd.Series(np.asarray(fitted.forecast(steps=len(future)), dtype=float), index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_dols_fmols(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["dols_fmols"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_dols_fmols(train)
        predicted = cointegration_regression_predict(fitted, future[exog_columns], future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_markov_switching(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["markov_switching"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_markov_switching(train)
        predicted = pd.Series(markov_switching_forecast_path(fitted, future[exog_columns]), index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_arimax_garch(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["arimax_garch"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    near_logpdf: list[float] = []
    far_logpdf: list[float] = []
    near_baseline_logpdf: list[float] = []
    far_baseline_logpdf: list[float] = []
    coverage_hits_90: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_arimax_garch(train["hpi"], build_exog(train, exog_columns))
        forecast = fitted.mean_result.get_forecast(steps=len(future), exog=build_exog(future, exog_columns))
        predicted = pd.Series(np.asarray(forecast.predicted_mean, dtype=float), index=future.index)
        sigma = pd.Series(arimax_garch_forecast_sigma(fitted, len(future)), index=future.index)
        delta_90 = pd.Series(arimax_garch_interval_delta(fitted, sigma.to_numpy(), alpha=0.10), index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)
        errors = future["hpi"].astype(float) - predicted
        model_logpdf = pd.Series(arimax_garch_logpdf(fitted, errors.to_numpy(), sigma.to_numpy()), index=future.index)
        benchmark_sigma_value = max(float(pd.Series(np.asarray(fitted.mean_result.resid, dtype=float)).dropna().std(ddof=0)), 1e-6)
        benchmark_sigma = pd.Series(np.repeat(benchmark_sigma_value, len(future)), index=future.index)
        baseline_logpdf = pd.Series(norm.logpdf(errors.to_numpy(), loc=0.0, scale=benchmark_sigma.to_numpy()), index=future.index)
        lower_90 = predicted - delta_90
        upper_90 = predicted + delta_90

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                    "y_sigma": float(sigma.loc[idx]),
                    "y_lower_90": float(lower_90.loc[idx]),
                    "y_upper_90": float(upper_90.loc[idx]),
                    "logpdf": float(model_logpdf.loc[idx]),
                    "logpdf_benchmark": float(baseline_logpdf.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())
        near_logpdf.extend(model_logpdf.loc[near_slice].tolist())
        far_logpdf.extend(model_logpdf.loc[far_slice].tolist())
        near_baseline_logpdf.extend(baseline_logpdf.loc[near_slice].tolist())
        far_baseline_logpdf.extend(baseline_logpdf.loc[far_slice].tolist())
        coverage_hits_90.extend(((future["hpi"] >= lower_90) & (future["hpi"] <= upper_90)).astype(float).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    model_near_nll = -float(np.sum(near_logpdf))
    model_far_nll = -float(np.sum(far_logpdf))
    baseline_near_nll = -float(np.sum(near_baseline_logpdf))
    baseline_far_nll = -float(np.sum(far_baseline_logpdf))
    gof_vol_near = 1.0 - (model_near_nll / baseline_near_nll if baseline_near_nll else np.inf)
    gof_vol_far = 1.0 - (model_far_nll / baseline_far_nll if baseline_far_nll else np.inf)
    coverage_90 = float(np.mean(coverage_hits_90)) if coverage_hits_90 else np.nan
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse, gof_vol_near, gof_vol_far, coverage_90


def walk_forward_validation_state_space(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    exog_columns = get_active_spec()["state_space"]["exogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_state_space(train["hpi"], build_exog(train, exog_columns))
        predicted_values, _, _ = state_space_forecast_components(
            fitted,
            steps=len(future),
            future_exog=build_exog(future, exog_columns),
        )
        predicted = pd.Series(predicted_values, index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_var(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    endog_columns = get_active_spec()["var"]["endogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_var(train)
        lag_order = int(fitted.k_ar)
        if len(train) <= lag_order:
            continue
        history = train[endog_columns].astype(float).to_numpy()[-lag_order:]
        forecast_values = fitted.forecast(history, steps=len(future))
        predicted = pd.Series(forecast_values[:, endog_columns.index("hpi")], index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_bvar(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    endog_columns = get_active_spec()["bvar"]["endogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_bvar(train)
        lag_order = int(fitted.k_ar)
        if len(train) <= lag_order:
            continue
        history = train[endog_columns].astype(float).to_numpy()[-lag_order:]
        forecast_values = bvar_forecast(fitted, history, steps=len(future))
        predicted = pd.Series(forecast_values[:, endog_columns.index("hpi")], index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_threshold_var(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    endog_columns = get_active_spec()["threshold_var"]["endogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_threshold_var(train)
        lag_order = int(fitted.k_ar)
        if len(train) <= lag_order:
            continue
        history = train[endog_columns].astype(float).to_numpy()[-lag_order:]
        forecast_values = threshold_var_forecast(fitted, history, steps=len(future))
        predicted = pd.Series(forecast_values[:, endog_columns.index("hpi")], index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_vecm(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    endog_columns = get_active_spec()["vecm"]["endogenous"]
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_vecm(train)
        forecast_values = fitted.predict(steps=len(future))
        predicted = pd.Series(forecast_values[:, endog_columns.index("hpi")], index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_dynamic_factor(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    endog_columns = get_active_spec()["dynamic_factor"]["endogenous"]
    hpi_idx = endog_columns.index("hpi")
    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue

        fitted = fit_dynamic_factor(train)
        predicted_frame, _, _ = dynamic_factor_forecast_components(fitted, endog_columns, steps=len(future))
        predicted = pd.Series(predicted_frame.iloc[:, hpi_idx].to_numpy(), index=future.index)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            horizon = int(idx - end_idx)
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": horizon,
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_regularized_linear(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["regularized_linear"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train = search_frame.iloc[: end_idx + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_regularized_linear(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = regularized_linear_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_random_forest(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["random_forest"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_random_forest(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = tabular_ml_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_gradient_boosting(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["gradient_boosting"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_gradient_boosting(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = tabular_ml_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_support_vector_regression(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["support_vector_regression"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_support_vector_regression(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = tabular_ml_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_xgboost(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["xgboost"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_xgboost(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = tabular_ml_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_lightgbm(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["lightgbm"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_lightgbm(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = tabular_ml_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_neuralprophet(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["neuralprophet"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_neuralprophet(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = neuralprophet_forecast_path(fitted, future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_lstm_gru(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["lstm_gru"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_lstm_gru(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = recurrent_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_tcn(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["tcn"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_tcn(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = recurrent_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_nbeats(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["nbeats"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_nbeats(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = recurrent_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def walk_forward_validation_transformer(
    search_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    if not origins:
        raise RuntimeError("No backtest origins were generated. Check the prepared sample length.")

    predictions: list[dict[str, Any]] = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    near_naive_errors: list[float] = []
    far_naive_errors: list[float] = []
    exog_columns = get_active_spec()["transformer"]["exogenous"]

    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        train = search_frame.iloc[: int(origin_idx[0]) + 1].copy().reset_index(drop=True)
        future = search_frame.iloc[int(origin_idx[0]) + 1 : int(origin_idx[0]) + 41].copy().reset_index(drop=True)
        if len(future) < 40:
            continue

        fitted = fit_transformer(train)
        future_exog = future[["date"] + exog_columns].copy() if exog_columns else future[["date"]].copy()
        predicted_values = recurrent_forecast_path(fitted, train[["date", "hpi"] + exog_columns], future_exog)
        predicted = pd.Series(predicted_values, index=future.index, dtype=float)
        naive = pd.Series(forecast_naive_path(train["hpi"], len(future)), index=future.index)

        for idx, row in future.iterrows():
            predictions.append(
                {
                    "experiment_id": "",
                    "trial_id": "",
                    "origin_date": origin,
                    "forecast_date": row["date"],
                    "horizon_q": int(idx + 1),
                    "y_true": float(row["hpi"]),
                    "y_pred": float(predicted.loc[idx]),
                }
            )

        near_slice = future.index[:12]
        far_slice = future.index[12:]
        near_errors.extend((future.loc[near_slice, "hpi"] - predicted.loc[near_slice]).tolist())
        far_errors.extend((future.loc[far_slice, "hpi"] - predicted.loc[far_slice]).tolist())
        near_naive_errors.extend((future.loc[near_slice, "hpi"] - naive.loc[near_slice]).tolist())
        far_naive_errors.extend((future.loc[far_slice, "hpi"] - naive.loc[far_slice]).tolist())

    pred_frame = pd.DataFrame(predictions)
    val_near_rmse = rmse(np.zeros(len(near_errors)), np.asarray(near_errors))
    val_far_rmse = rmse(np.zeros(len(far_errors)), np.asarray(far_errors))
    naive_near_rmse = rmse(np.zeros(len(near_naive_errors)), np.asarray(near_naive_errors))
    naive_far_rmse = rmse(np.zeros(len(far_naive_errors)), np.asarray(far_naive_errors))
    gof_near = 1.0 - (val_near_rmse / naive_near_rmse if naive_near_rmse else np.inf)
    gof_far = 1.0 - (val_far_rmse / naive_far_rmse if naive_far_rmse else np.inf)
    return pred_frame, gof_near, gof_far, val_near_rmse, val_far_rmse, naive_near_rmse


def in_sample_metrics(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    exog_columns = get_active_spec()["arimax"]["exogenous"]
    fitted = fit_arimax(frame["hpi"], build_exog(frame, exog_columns))
    predicted = pd.Series(np.asarray(fitted.fittedvalues), index=frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_ardl(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_ardl(frame)
    predicted = pd.Series(np.asarray(fitted.fittedvalues, dtype=float), index=fitted.fittedvalues.index)
    aligned = pd.DataFrame({"y": frame.loc[predicted.index, "hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_uecm(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_uecm(frame)
    predicted = pd.Series(np.asarray(fitted.fittedvalues, dtype=float), index=fitted.fittedvalues.index)
    aligned = pd.DataFrame({"y": frame.loc[predicted.index, "hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_ets(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_ets(frame)
    predicted = pd.Series(np.asarray(fitted.fittedvalues, dtype=float), index=frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_dols_fmols(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_dols_fmols(frame)
    aligned = pd.DataFrame({"y": frame.loc[fitted.fittedvalues.index, "hpi"], "pred": fitted.fittedvalues}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame.loc[aligned.index, "hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_markov_switching(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_markov_switching(frame)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": fitted.fittedvalues}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_arimax_garch(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    exog_columns = get_active_spec()["arimax_garch"]["exogenous"]
    fitted = fit_arimax_garch(frame["hpi"], build_exog(frame, exog_columns))
    predicted = pd.Series(np.asarray(fitted.mean_result.fittedvalues, dtype=float), index=frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_state_space(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    exog_columns = get_active_spec()["state_space"]["exogenous"]
    fitted = fit_state_space(frame["hpi"], build_exog(frame, exog_columns))
    predicted = state_space_fittedvalues_series(fitted, frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_var(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_var(frame)
    predicted = pd.Series(np.asarray(fitted.fittedvalues["hpi"]), index=fitted.fittedvalues.index)
    aligned = pd.DataFrame({"y": frame.loc[predicted.index, "hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_bvar(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_bvar(frame)
    predicted = pd.Series(np.asarray(fitted.fittedvalues["hpi"]), index=fitted.fittedvalues.index)
    aligned = pd.DataFrame({"y": frame.loc[predicted.index, "hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_threshold_var(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_threshold_var(frame)
    predicted = pd.Series(np.asarray(fitted.fittedvalues["hpi"]), index=fitted.fittedvalues.index)
    aligned = pd.DataFrame({"y": frame.loc[predicted.index, "hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_vecm(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_vecm(frame)
    endog_columns = get_active_spec()["vecm"]["endogenous"]
    start_idx = len(frame) - len(fitted.fittedvalues)
    fitted_index = frame.index[start_idx:]
    hpi_position = endog_columns.index("hpi")
    predicted = pd.Series(np.asarray(fitted.fittedvalues)[:, hpi_position], index=fitted_index)
    aligned = pd.DataFrame({"y": frame.loc[predicted.index, "hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_dynamic_factor(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_dynamic_factor(frame)
    endog_columns = get_active_spec()["dynamic_factor"]["endogenous"]
    hpi_idx = endog_columns.index("hpi")
    predicted_frame = dynamic_factor_fittedvalues_frame(fitted, endog_columns, frame.index)
    predicted = pd.Series(predicted_frame.iloc[:, hpi_idx].to_numpy(), index=frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_regularized_linear(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_regularized_linear(frame)
    target_lags = [int(lag) for lag in get_active_spec()["regularized_linear"]["target_lags"]]
    max_lag = max(target_lags)
    aligned_y = frame["hpi"].iloc[max_lag:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_random_forest(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_random_forest(frame)
    target_lags = [int(lag) for lag in get_active_spec()["random_forest"]["target_lags"]]
    max_lag = max(target_lags)
    aligned_y = frame["hpi"].iloc[max_lag:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_gradient_boosting(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_gradient_boosting(frame)
    target_lags = [int(lag) for lag in get_active_spec()["gradient_boosting"]["target_lags"]]
    max_lag = max(target_lags)
    aligned_y = frame["hpi"].iloc[max_lag:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_support_vector_regression(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_support_vector_regression(frame)
    target_lags = [int(lag) for lag in get_active_spec()["support_vector_regression"]["target_lags"]]
    max_lag = max(target_lags)
    aligned_y = frame["hpi"].iloc[max_lag:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_xgboost(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_xgboost(frame)
    target_lags = [int(lag) for lag in get_active_spec()["xgboost"]["target_lags"]]
    max_lag = max(target_lags)
    aligned_y = frame["hpi"].iloc[max_lag:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_lightgbm(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_lightgbm(frame)
    target_lags = [int(lag) for lag in get_active_spec()["lightgbm"]["target_lags"]]
    max_lag = max(target_lags)
    aligned_y = frame["hpi"].iloc[max_lag:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_neuralprophet(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_neuralprophet(frame)
    target_lags = int(get_active_spec()["neuralprophet"]["n_lags"])
    aligned_y = frame["hpi"].iloc[target_lags:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_lstm_gru(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_lstm_gru(frame)
    lookback = int(get_active_spec()["lstm_gru"]["lookback"])
    aligned_y = frame["hpi"].iloc[lookback:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_tcn(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_tcn(frame)
    lookback = int(get_active_spec()["tcn"]["lookback"])
    aligned_y = frame["hpi"].iloc[lookback:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_nbeats(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_nbeats(frame)
    lookback = int(get_active_spec()["nbeats"]["lookback"])
    aligned_y = frame["hpi"].iloc[lookback:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def in_sample_metrics_transformer(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    fitted = fit_transformer(frame)
    lookback = int(get_active_spec()["transformer"]["lookback"])
    aligned_y = frame["hpi"].iloc[lookback:].reset_index(drop=True)
    predicted = fitted.fittedvalues.reset_index(drop=True)
    aligned = pd.DataFrame({"y": aligned_y, "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def model_n_params(fitted: Any) -> int:
    if hasattr(fitted, "params"):
        return int(np.size(np.asarray(fitted.params)))

    total = 0
    for attr in ["alpha", "beta", "gamma", "det_coef", "det_coef_coint"]:
        value = getattr(fitted, attr, None)
        if value is not None:
            total += int(np.size(np.asarray(value)))
    return total


def model_param_lookup(fitted: Any) -> dict[str, float]:
    if isinstance(fitted, ArimaxGarchFit):
        combined = {}
        combined.update(model_param_lookup(fitted.mean_result))
        combined.update(model_param_lookup(fitted.volatility_result))
        return combined

    params = getattr(fitted, "params", None)
    if params is None:
        return {}

    if hasattr(params, "to_dict"):
        return {str(key): float(value) for key, value in params.to_dict().items()}

    names = getattr(fitted, "param_names", None)
    values = np.asarray(params, dtype=float).reshape(-1)
    if names is not None and len(names) == len(values):
        return {str(name): float(value) for name, value in zip(names, values)}
    return {}


def coefficient_for_variable(fitted: Any, variable: str) -> float:
    lookup = model_param_lookup(fitted)
    if variable in lookup:
        return lookup[variable]
    for name, value in lookup.items():
        if name.endswith(f".{variable}") or name.endswith(f"_{variable}") or variable in name:
            return value
    return 0.0


def diagnostic_score(
    fitted,
    residuals: pd.Series,
    future_forecast: pd.Series,
    exog_columns: list[str],
    variable_config: dict[str, Any],
    extra_checks: dict[str, bool] | None = None,
) -> tuple[float, list[str], list[str]]:
    passes: list[str] = []
    fails: list[str] = []
    resids = residuals.dropna()

    try:
        if float(acorr_ljungbox(resids, lags=[12], return_df=True)["lb_pvalue"].iloc[0]) > 0.05:
            passes.append("ljung_box")
        else:
            fails.append("ljung_box")
    except Exception:
        fails.append("ljung_box")

    try:
        if float(jarque_bera(resids).pvalue) > 0.01:
            passes.append("jarque_bera")
        else:
            fails.append("jarque_bera")
    except Exception:
        fails.append("jarque_bera")

    try:
        if float(het_arch(resids, nlags=4)[1]) > 0.05:
            passes.append("arch_lm")
        else:
            fails.append("arch_lm")
    except Exception:
        fails.append("arch_lm")

    try:
        if float(breaks_cusumolsresid(resids, ddof=model_n_params(fitted))[1]) > 0.05:
            passes.append("cusum")
        else:
            fails.append("cusum")
    except Exception:
        fails.append("cusum")

    try:
        if float(adfuller(resids)[1]) < 0.05:
            passes.append("adf")
        else:
            fails.append("adf")
    except Exception:
        fails.append("adf")

    sign_mapping = expected_signs(variable_config)
    if exog_columns:
        sign_fail = False
        for variable in exog_columns:
            expected = sign_mapping.get(variable)
            if not expected:
                continue
            actual = coefficient_for_variable(fitted, variable)
            if expected == "positive" and actual <= 0:
                sign_fail = True
            if expected == "negative" and actual >= 0:
                sign_fail = True
        if sign_fail:
            fails.append("economic_sign")
        else:
            passes.append("economic_sign")

    changes = future_forecast.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    start_value = float(future_forecast.iloc[0]) if len(future_forecast) else np.nan
    end_value = float(future_forecast.iloc[-1]) if len(future_forecast) else np.nan
    if len(future_forecast) > 1 and np.isfinite(start_value) and start_value != 0.0:
        cumulative = (end_value / start_value) - 1.0
    else:
        cumulative = np.nan
    plausibility_ok = (
        (future_forecast > 0).all()
        and (changes.abs() < 0.15).all()
        and np.isfinite(cumulative)
        and (-0.20 <= cumulative <= 5.0)
        and (float(future_forecast.std()) > 0.0)
    )
    if plausibility_ok:
        passes.append("forecast_plausibility")
    else:
        fails.append("forecast_plausibility")

    for name, passed_check in (extra_checks or {}).items():
        if passed_check:
            passes.append(name)
        else:
            fails.append(name)

    total_checks = len(passes) + len(fails)
    score = len(passes) / total_checks if total_checks else 0.0
    return score, passes, fails


def build_final_forecast(search_frame: pd.DataFrame, variable_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    exog_columns = get_active_spec()["arimax"]["exogenous"]
    fitted = fit_arimax(search_frame["hpi"], build_exog(search_frame, exog_columns))
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    forecast = fitted.get_forecast(steps=120, exog=build_exog(future_exog, exog_columns))
    conf_90 = forecast.conf_int(alpha=0.10)
    conf_50 = forecast.conf_int(alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": np.asarray(forecast.predicted_mean),
            "hpi_lower_90": np.asarray(conf_90.iloc[:, 0]),
            "hpi_upper_90": np.asarray(conf_90.iloc[:, 1]),
            "hpi_lower_50": np.asarray(conf_50.iloc[:, 0]),
            "hpi_upper_50": np.asarray(conf_50.iloc[:, 1]),
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_ardl(search_frame: pd.DataFrame, variable_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    exog_columns = get_active_spec()["ardl"]["exogenous"]
    fitted = fit_ardl(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    forecast = np.asarray(fitted.forecast(steps=120, exog=build_exog(future_exog, exog_columns)), dtype=float)
    residual_std = max(float(pd.Series(np.asarray(fitted.resid, dtype=float)).dropna().std(ddof=0)), 1e-6)
    lower_90, upper_90 = regularized_linear_interval(forecast, residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_uecm(search_frame: pd.DataFrame, variable_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    exog_columns = get_active_spec()["uecm"]["exogenous"]
    fitted = fit_uecm(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    forecast = uecm_forecast_path(
        fitted,
        search_frame["hpi"],
        search_frame[exog_columns],
        future_exog[exog_columns],
    )
    residual_std = max(float(pd.Series(np.asarray(fitted.resid, dtype=float)).dropna().std(ddof=0)), 1e-6)
    lower_90, upper_90 = regularized_linear_interval(forecast, residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_ets(search_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    fitted = fit_ets(search_frame)
    forecast = np.asarray(fitted.forecast(steps=120), dtype=float)
    residual_std = max(float(pd.Series(np.asarray(fitted.resid, dtype=float)).dropna().std(ddof=0)), 1e-6)
    lower_90, upper_90 = regularized_linear_interval(forecast, residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, residual_std, alpha=0.50)
    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=120, freq="QE-DEC")

    final = pd.DataFrame(
        {
            "date": future_dates,
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_dols_fmols(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["dols_fmols"]["exogenous"]
    fitted = fit_dols_fmols(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    forecast = cointegration_regression_predict(fitted, future_exog[exog_columns], future_exog.index).to_numpy(dtype=float)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_markov_switching(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["markov_switching"]["exogenous"]
    fitted = fit_markov_switching(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    forecast = markov_switching_forecast_path(fitted, future_exog[exog_columns])
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_arimax_garch(search_frame: pd.DataFrame, variable_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    exog_columns = get_active_spec()["arimax_garch"]["exogenous"]
    fitted = fit_arimax_garch(search_frame["hpi"], build_exog(search_frame, exog_columns))
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    forecast = fitted.mean_result.get_forecast(steps=120, exog=build_exog(future_exog, exog_columns))
    mean_forecast = np.asarray(forecast.predicted_mean, dtype=float)
    sigma = arimax_garch_forecast_sigma(fitted, 120)
    delta_90 = arimax_garch_interval_delta(fitted, sigma, alpha=0.10)
    delta_50 = arimax_garch_interval_delta(fitted, sigma, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": mean_forecast,
            "hpi_lower_90": mean_forecast - delta_90,
            "hpi_upper_90": mean_forecast + delta_90,
            "hpi_lower_50": mean_forecast - delta_50,
            "hpi_upper_50": mean_forecast + delta_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_state_space(search_frame: pd.DataFrame, variable_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    exog_columns = get_active_spec()["state_space"]["exogenous"]
    fitted = fit_state_space(search_frame["hpi"], build_exog(search_frame, exog_columns))
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    predicted, lower_90, upper_90 = state_space_forecast_components(
        fitted,
        steps=120,
        future_exog=build_exog(future_exog, exog_columns),
        alpha=0.10,
    )
    _, lower_50, upper_50 = state_space_forecast_components(
        fitted,
        steps=120,
        future_exog=build_exog(future_exog, exog_columns),
        alpha=0.50,
    )
    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": predicted,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_var(search_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    spec = get_active_spec()["var"]
    endog_columns = spec["endogenous"]
    fitted = fit_var(search_frame)
    lag_order = int(fitted.k_ar)
    history = search_frame[endog_columns].astype(float).to_numpy()[-lag_order:]
    forecast = fitted.forecast_interval(history, steps=120, alpha=0.10)
    forecast_90, lower_90, upper_90 = forecast
    forecast_50, lower_50, upper_50 = fitted.forecast_interval(history, steps=120, alpha=0.50)

    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=120, freq="QE-DEC")
    hpi_idx = endog_columns.index("hpi")
    final = pd.DataFrame(
        {
            "date": future_dates,
            "hpi_actual": np.nan,
            "hpi_forecast": forecast_90[:, hpi_idx],
            "hpi_lower_90": lower_90[:, hpi_idx],
            "hpi_upper_90": upper_90[:, hpi_idx],
            "hpi_lower_50": lower_50[:, hpi_idx],
            "hpi_upper_50": upper_50[:, hpi_idx],
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_bvar(search_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    spec = get_active_spec()["bvar"]
    endog_columns = spec["endogenous"]
    fitted = fit_bvar(search_frame)
    lag_order = int(fitted.k_ar)
    history = search_frame[endog_columns].astype(float).to_numpy()[-lag_order:]
    forecast_90, lower_90, upper_90 = bvar_forecast_interval(fitted, history, steps=120, alpha=0.10)
    forecast_50, lower_50, upper_50 = bvar_forecast_interval(fitted, history, steps=120, alpha=0.50)

    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=120, freq="QE-DEC")
    hpi_idx = endog_columns.index("hpi")
    final = pd.DataFrame(
        {
            "date": future_dates,
            "hpi_actual": np.nan,
            "hpi_forecast": forecast_90[:, hpi_idx],
            "hpi_lower_90": lower_90[:, hpi_idx],
            "hpi_upper_90": upper_90[:, hpi_idx],
            "hpi_lower_50": lower_50[:, hpi_idx],
            "hpi_upper_50": upper_50[:, hpi_idx],
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_threshold_var(search_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    spec = get_active_spec()["threshold_var"]
    endog_columns = spec["endogenous"]
    fitted = fit_threshold_var(search_frame)
    lag_order = int(fitted.k_ar)
    history = search_frame[endog_columns].astype(float).to_numpy()[-lag_order:]
    forecast_90, lower_90, upper_90 = threshold_var_forecast_interval(fitted, history, steps=120, alpha=0.10)
    forecast_50, lower_50, upper_50 = threshold_var_forecast_interval(fitted, history, steps=120, alpha=0.50)

    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=120, freq="QE-DEC")
    hpi_idx = endog_columns.index("hpi")
    final = pd.DataFrame(
        {
            "date": future_dates,
            "hpi_actual": np.nan,
            "hpi_forecast": forecast_90[:, hpi_idx],
            "hpi_lower_90": lower_90[:, hpi_idx],
            "hpi_upper_90": upper_90[:, hpi_idx],
            "hpi_lower_50": lower_50[:, hpi_idx],
            "hpi_upper_50": upper_50[:, hpi_idx],
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_vecm(search_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    spec = get_active_spec()["vecm"]
    endog_columns = spec["endogenous"]
    fitted = fit_vecm(search_frame)
    forecast_90, lower_90, upper_90 = fitted.predict(steps=120, alpha=0.10)
    forecast_50, lower_50, upper_50 = fitted.predict(steps=120, alpha=0.50)

    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=120, freq="QE-DEC")
    hpi_idx = endog_columns.index("hpi")
    final = pd.DataFrame(
        {
            "date": future_dates,
            "hpi_actual": np.nan,
            "hpi_forecast": forecast_90[:, hpi_idx],
            "hpi_lower_90": lower_90[:, hpi_idx],
            "hpi_upper_90": upper_90[:, hpi_idx],
            "hpi_lower_50": lower_50[:, hpi_idx],
            "hpi_upper_50": upper_50[:, hpi_idx],
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_dynamic_factor(search_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    spec = get_active_spec()["dynamic_factor"]
    endog_columns = spec["endogenous"]
    hpi_idx = endog_columns.index("hpi")
    fitted = fit_dynamic_factor(search_frame)
    predicted_frame, lower_90, upper_90 = dynamic_factor_forecast_components(fitted, endog_columns, steps=120, alpha=0.10)
    _, lower_50, upper_50 = dynamic_factor_forecast_components(fitted, endog_columns, steps=120, alpha=0.50)

    future_dates = pd.date_range(search_frame["date"].iloc[-1] + pd.offsets.QuarterEnd(), periods=120, freq="QE-DEC")
    final = pd.DataFrame(
        {
            "date": future_dates,
            "hpi_actual": np.nan,
            "hpi_forecast": predicted_frame.iloc[:, hpi_idx].to_numpy(),
            "hpi_lower_90": lower_90.iloc[:, hpi_idx].to_numpy(),
            "hpi_upper_90": upper_90.iloc[:, hpi_idx].to_numpy(),
            "hpi_lower_50": lower_50.iloc[:, hpi_idx].to_numpy(),
            "hpi_upper_50": upper_50.iloc[:, hpi_idx].to_numpy(),
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_regularized_linear(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["regularized_linear"]["exogenous"]
    fitted = fit_regularized_linear(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = regularized_linear_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_random_forest(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["random_forest"]["exogenous"]
    fitted = fit_random_forest(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = tabular_ml_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_gradient_boosting(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["gradient_boosting"]["exogenous"]
    fitted = fit_gradient_boosting(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = tabular_ml_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_support_vector_regression(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["support_vector_regression"]["exogenous"]
    fitted = fit_support_vector_regression(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = tabular_ml_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_xgboost(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["xgboost"]["exogenous"]
    fitted = fit_xgboost(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = tabular_ml_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_lightgbm(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["lightgbm"]["exogenous"]
    fitted = fit_lightgbm(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = tabular_ml_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_neuralprophet(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["neuralprophet"]["exogenous"]
    fitted = fit_neuralprophet(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = neuralprophet_forecast_path(fitted, future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_lstm_gru(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["lstm_gru"]["exogenous"]
    fitted = fit_lstm_gru(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = recurrent_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_tcn(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["tcn"]["exogenous"]
    fitted = fit_tcn(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = recurrent_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_nbeats(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["nbeats"]["exogenous"]
    fitted = fit_nbeats(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = recurrent_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def build_final_forecast_transformer(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    exog_columns = get_active_spec()["transformer"]["exogenous"]
    fitted = fit_transformer(search_frame)
    future_exog = build_future_exog(search_frame, variable_config, horizon=120)
    future_inputs = future_exog[["date"] + exog_columns].copy() if exog_columns else future_exog[["date"]].copy()
    forecast = recurrent_forecast_path(fitted, search_frame[["date", "hpi"] + exog_columns], future_inputs)
    lower_90, upper_90 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = regularized_linear_interval(forecast, fitted.residual_std, alpha=0.50)

    final = pd.DataFrame(
        {
            "date": future_exog["date"],
            "hpi_actual": np.nan,
            "hpi_forecast": forecast,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )
    final.to_csv(output_dir / "forecast_120q.csv", index=False)
    return final


def write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)


def append_tsv(path: Path, header: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row], columns=header)
    if path.exists():
        frame.to_csv(path, sep="\t", index=False, header=False, mode="a")
    else:
        frame.to_csv(path, sep="\t", index=False)


def upsert_leaderboard(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        board = pd.read_csv(path, sep="\t")
    else:
        board = pd.DataFrame(columns=LEADERBOARD_HEADER)

    board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
    board["gof_composite"] = pd.to_numeric(board["gof_composite"], errors="coerce")
    board["n_trials"] = pd.to_numeric(board["n_trials"], errors="coerce").fillna(0).astype(int)
    max_trials_by_experiment = board.groupby("experiment_id")["n_trials"].max().to_dict()
    board = board.sort_values("gof_composite", ascending=False).drop_duplicates("experiment_id", keep="first").reset_index(drop=True)
    board["n_trials"] = board["experiment_id"].map(max_trials_by_experiment).fillna(board["n_trials"]).astype(int)
    board["rank"] = np.arange(1, len(board) + 1)
    board.to_csv(path, sep="\t", index=False)


def build_results_row(metrics: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "trial_id": metrics["trial_id"],
        "accepted_commit": "",
        "model_class": metrics["model_class"],
        "champion_eligible": metrics["champion_eligible"],
        "gof_composite": metrics["gof_composite"],
        "gof_insample": metrics["gof_insample"],
        "gof_val_near": metrics["gof_validation_near"],
        "gof_val_far": metrics["gof_validation_far"],
        "gof_diag": metrics["gof_diagnostic"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "theil_u": metrics["theil_u"],
        "n_params": metrics["n_params"],
        "status": metrics["status"],
        "description": metrics["description"],
        "artifact_dir": str(output_dir),
        "error_summary": metrics["error_summary"],
    }


def build_leaderboard_row(metrics: dict[str, Any], trial_count: int) -> dict[str, Any]:
    return {
        "rank": 1,
        "experiment_id": metrics["experiment_id"],
        "model_class": metrics["model_class"],
        "champion_eligible": metrics["champion_eligible"],
        "best_trial_id": metrics["trial_id"],
        "best_commit": git_commit_short(),
        "gof_composite": metrics["gof_composite"],
        "gof_insample": metrics["gof_insample"],
        "gof_val_near": metrics["gof_validation_near"],
        "gof_val_far": metrics["gof_validation_far"],
        "rmse_1yr": metrics["rmse_1yr"],
        "rmse_3yr": metrics["rmse_3yr"],
        "n_params": metrics["n_params"],
        "n_trials": trial_count,
        "description": metrics["description"],
    }


def execute_search(args: argparse.Namespace, output_dir: Path, logger: TrialLogger) -> dict[str, Any]:
    search_frame, _, manifest, variable_config = load_panels()
    search_frame = filter_sample(search_frame)
    logger.info(f"Loaded {len(search_frame)} search rows for {args.model_class}.")

    if args.model_class == "ARIMAX":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation(search_frame, manifest)
        future_forecast = build_final_forecast(search_frame, variable_config, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid))
        sign_columns = get_active_spec()["arimax"]["exogenous"]
        extra_diag_checks: dict[str, bool] = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "ARDL":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_ardl(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_ardl(search_frame, manifest)
        future_forecast = build_final_forecast_ardl(search_frame, variable_config, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid, dtype=float), index=fitted.resid.index)
        sign_columns = get_active_spec()["ardl"]["exogenous"]
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "UECM / ECM":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_uecm(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_uecm(search_frame, manifest)
        future_forecast = build_final_forecast_uecm(search_frame, variable_config, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid, dtype=float), index=fitted.resid.index)
        sign_columns = get_active_spec()["uecm"]["exogenous"]
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "DOLS / FMOLS":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_dols_fmols(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_dols_fmols(search_frame, manifest)
        future_forecast = build_final_forecast_dols_fmols(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = get_active_spec()["dols_fmols"]["exogenous"]
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Markov-Switching AR / ARX":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_markov_switching(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_markov_switching(search_frame, manifest)
        future_forecast = build_final_forecast_markov_switching(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = get_active_spec()["markov_switching"]["exogenous"]
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "ETS":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_ets(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_ets(search_frame, manifest)
        future_forecast = build_final_forecast_ets(search_frame, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid, dtype=float), index=search_frame.index)
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "ARIMAX-GARCH":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_arimax_garch(search_frame)
        (
            validation_frame,
            gof_near,
            gof_far,
            rmse_1yr,
            rmse_3yr,
            naive_rmse_1yr,
            gof_vol_near,
            gof_vol_far,
            coverage_90,
        ) = walk_forward_validation_arimax_garch(search_frame, manifest)
        future_forecast = build_final_forecast_arimax_garch(search_frame, variable_config, output_dir)
        residuals = arimax_garch_standardized_residuals(fitted)
        sign_columns = get_active_spec()["arimax_garch"]["exogenous"]
        volatility_weighted_score = 0.7 * gof_vol_near + 0.3 * gof_vol_far
        residual_variance = float(residuals.var(ddof=0)) if not residuals.empty else np.nan
        extra_diag_checks = {
            "volatility_coverage_90": bool(np.isfinite(coverage_90) and 0.80 <= coverage_90 <= 0.98),
            "std_resid_unit_variance": bool(np.isfinite(residual_variance) and 0.8 <= residual_variance <= 1.2),
        }
    elif args.model_class == "State-Space":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_state_space(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_state_space(search_frame, manifest)
        future_forecast = build_final_forecast_state_space(search_frame, variable_config, output_dir)
        residuals = state_space_residuals_series(fitted, search_frame["hpi"])
        sign_columns = get_active_spec()["state_space"]["exogenous"]
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "VAR":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_var(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_var(search_frame, manifest)
        future_forecast = build_final_forecast_var(search_frame, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid["hpi"]))
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "BVAR":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_bvar(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_bvar(search_frame, manifest)
        future_forecast = build_final_forecast_bvar(search_frame, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid["hpi"]))
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class in {"Threshold VAR", "Threshold VAR / SETAR"}:
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_threshold_var(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_threshold_var(search_frame, manifest)
        future_forecast = build_final_forecast_threshold_var(search_frame, output_dir)
        residuals = pd.Series(np.asarray(fitted.resid["hpi"]))
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "VECM":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_vecm(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_vecm(search_frame, manifest)
        future_forecast = build_final_forecast_vecm(search_frame, output_dir)
        endog_columns = get_active_spec()["vecm"]["endogenous"]
        residuals = pd.Series(np.asarray(fitted.resid)[:, endog_columns.index("hpi")])
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Dynamic Factor":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_dynamic_factor(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_dynamic_factor(search_frame, manifest)
        future_forecast = build_final_forecast_dynamic_factor(search_frame, output_dir)
        endog_columns = get_active_spec()["dynamic_factor"]["endogenous"]
        residuals = pd.Series(np.asarray(fitted.resid, dtype=float)[:, endog_columns.index("hpi")])
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Ridge / Lasso / Elastic Net":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_regularized_linear(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_regularized_linear(search_frame, manifest)
        future_forecast = build_final_forecast_regularized_linear(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = get_active_spec()["regularized_linear"]["exogenous"]
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Random Forest":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_random_forest(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_random_forest(search_frame, manifest)
        future_forecast = build_final_forecast_random_forest(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Gradient Boosting":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_gradient_boosting(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_gradient_boosting(search_frame, manifest)
        future_forecast = build_final_forecast_gradient_boosting(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Support Vector Regression":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_support_vector_regression(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_support_vector_regression(search_frame, manifest)
        future_forecast = build_final_forecast_support_vector_regression(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "XGBoost":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_xgboost(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_xgboost(search_frame, manifest)
        future_forecast = build_final_forecast_xgboost(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "LightGBM":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_lightgbm(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_lightgbm(search_frame, manifest)
        future_forecast = build_final_forecast_lightgbm(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "NeuralProphet":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_neuralprophet(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_neuralprophet(search_frame, manifest)
        future_forecast = build_final_forecast_neuralprophet(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "LSTM / GRU":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_lstm_gru(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_lstm_gru(search_frame, manifest)
        future_forecast = build_final_forecast_lstm_gru(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "TCN":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_tcn(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_tcn(search_frame, manifest)
        future_forecast = build_final_forecast_tcn(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "N-BEATS":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_nbeats(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_nbeats(search_frame, manifest)
        future_forecast = build_final_forecast_nbeats(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    elif args.model_class == "Transformer":
        fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics_transformer(search_frame)
        validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation_transformer(search_frame, manifest)
        future_forecast = build_final_forecast_transformer(search_frame, variable_config, output_dir)
        residuals = fitted.resid
        sign_columns = []
        extra_diag_checks = {}
        volatility_weighted_score = np.nan
    else:
        raise NotImplementedError(f"{args.model_class} is not implemented yet.")

    validation_frame["experiment_id"] = args.experiment_id
    validation_frame["trial_id"] = args.trial_id
    validation_frame.to_parquet(output_dir / "validation_predictions.parquet", index=False)

    gof_diag, passed, failed = diagnostic_score(
        fitted,
        residuals,
        future_forecast["hpi_forecast"],
        sign_columns,
        variable_config,
        extra_checks=extra_diag_checks,
    )

    if args.model_class == "ARIMAX-GARCH":
        gof_composite = 0.30 * gof_insample + 0.25 * gof_near + 0.10 * gof_far + 0.25 * volatility_weighted_score + 0.10 * gof_diag
    else:
        gof_composite = 0.40 * gof_insample + 0.35 * gof_near + 0.15 * gof_far + 0.10 * gof_diag
    overall_rmse = rmse_1yr
    overall_mae = mae(
        validation_frame.loc[validation_frame["horizon_q"] <= 12, "y_true"],
        validation_frame.loc[validation_frame["horizon_q"] <= 12, "y_pred"],
    )
    theil_u = overall_rmse / naive_rmse_1yr if naive_rmse_1yr else np.nan

    metrics = {
        "experiment_id": args.experiment_id,
        "trial_id": args.trial_id,
        "model_class": args.model_class,
        "champion_eligible": bool(get_active_spec()["champion_eligible"]),
        "description": get_active_spec()["description"],
        "gof_composite": float(gof_composite),
        "gof_insample": float(gof_insample),
        "gof_validation_near": float(gof_near),
        "gof_validation_far": float(gof_far),
        "gof_diagnostic": float(gof_diag),
        "rmse": float(overall_rmse),
        "mae": float(overall_mae),
        "theil_u": float(theil_u),
        "directional_accuracy": float(
            (
                np.sign(validation_frame["y_true"].diff().dropna())
                == np.sign(validation_frame["y_pred"].diff().dropna())
            ).mean()
        ),
        "n_params": model_n_params(fitted),
        "status": "ok",
        "diagnostics_passed": passed,
        "diagnostics_failed": failed,
        "error_summary": "",
        "rmse_1yr": float(rmse_1yr),
        "rmse_3yr": float(rmse_3yr),
        "insample_rmse": float(insample_rmse),
        "naive_insample_rmse": float(naive_insample_rmse),
        "gof_volatility_near": float(gof_vol_near) if args.model_class == "ARIMAX-GARCH" else np.nan,
        "gof_volatility_far": float(gof_vol_far) if args.model_class == "ARIMAX-GARCH" else np.nan,
        "gof_volatility_weighted": float(volatility_weighted_score) if args.model_class == "ARIMAX-GARCH" else np.nan,
        "validation_coverage_90": float(coverage_90) if args.model_class == "ARIMAX-GARCH" else np.nan,
        "fit_attempts": getattr(fitted, "_fit_attempts", []),
        "fit_strategy": getattr(fitted, "_fit_strategy", ""),
        "fit_scaled": bool(getattr(fitted, "_fit_scaled", False)),
        "fit_recovered": bool(getattr(fitted, "_fit_recovered", False)),
    }
    logger.info(f"Completed trial {args.trial_id} with GOF {metrics['gof_composite']:.4f}.")
    return metrics


def execute_finalize(args: argparse.Namespace, output_dir: Path, logger: TrialLogger) -> dict[str, Any]:
    metrics = execute_search(args, output_dir, logger)
    holdout = pd.read_parquet(HOLDOUT_PANEL_PATH)
    holdout["date"] = pd.to_datetime(holdout["date"])
    metrics["holdout_rows"] = int(len(holdout))
    return metrics


def persist_logs(metrics: dict[str, Any], output_dir: Path) -> None:
    experiment_dir = output_dir.parent.parent
    local_results = experiment_dir / "results.tsv"
    local_leaderboard = experiment_dir / "leaderboard.tsv"

    results_row = build_results_row(metrics, output_dir)
    append_tsv(local_results, RESULTS_HEADER, results_row)
    append_tsv(ROOT_RESULTS_PATH, RESULTS_HEADER, results_row)

    if metrics["status"] == "ok":
        trial_count = len(pd.read_csv(local_results, sep="\t"))
        leaderboard_row = build_leaderboard_row(metrics, trial_count)
        upsert_leaderboard(local_leaderboard, leaderboard_row)
        upsert_leaderboard(ROOT_LEADERBOARD_PATH, leaderboard_row)


def existing_trial_count(experiment_dir: Path) -> int:
    results_path = experiment_dir / "results.tsv"
    if not results_path.exists():
        return 0
    return len(pd.read_csv(results_path, sep="\t"))


def refinement_stage(experiment_id: str) -> int:
    lowered = experiment_id.lower()
    if "refine2" in lowered:
        return 2
    if "refine" in lowered:
        return 1
    return 0


def candidate_cap_for_experiment(model_class: str, refinement: int, max_trials: int) -> int:
    if refinement > 0:
        return min(max_trials, 60)

    expensive_model_classes = {
        "BVAR",
        "Threshold VAR",
        "Threshold VAR / SETAR",
        "NeuralProphet",
        "LSTM / GRU",
        "TCN",
        "N-BEATS",
        "Transformer",
    }
    if model_class in expensive_model_classes:
        return min(max_trials, 180)

    return max_trials


def build_arimax_spec(order: tuple[int, int, int], exogenous: list[str], trend: str = "t") -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"ARIMAX{order} trend={trend} with {label}",
        "arimax": {
            "order": list(order),
            "trend": trend,
            "exogenous": exogenous,
            "include_intercept": False,
            "enforce_stationarity": False,
            "enforce_invertibility": False,
            "maxiter": 200,
        },
    }


def build_arimax_garch_spec(
    order: tuple[int, int, int],
    exogenous: list[str],
    volatility: str,
    p: int,
    q: int,
    trend: str = "t",
    o: int = 0,
    distribution: str = "normal",
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    variance_label = f"{volatility}({p},{q})" if volatility == "GARCH" else f"{volatility}({p},{o},{q})"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"ARIMAX-GARCH{order} trend={trend} {variance_label} dist={distribution} with {label}",
        "arimax_garch": {
            "order": list(order),
            "trend": trend,
            "exogenous": exogenous,
            "include_intercept": False,
            "enforce_stationarity": False,
            "enforce_invertibility": False,
            "maxiter": 200,
            "volatility": volatility,
            "p": p,
            "o": o,
            "q": q,
            "distribution": distribution,
            "residual_scale": 100.0,
        },
    }


def build_ardl_spec(
    exogenous: list[str],
    ar_lags: int,
    distributed_lags: int,
    trend: str = "c",
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"ARDL(ar_lags={ar_lags}, distributed_lags={distributed_lags}) trend={trend} with {label}",
        "ardl": {
            "exogenous": exogenous,
            "ar_lags": ar_lags,
            "distributed_lags": distributed_lags,
            "trend": trend,
        },
    }


def build_uecm_spec(
    exogenous: list[str],
    ar_lags: int,
    distributed_lags: int,
    trend: str = "c",
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"UECM(ar_lags={ar_lags}, distributed_lags={distributed_lags}) trend={trend} with {label}",
        "uecm": {
            "exogenous": exogenous,
            "ar_lags": ar_lags,
            "distributed_lags": distributed_lags,
            "trend": trend,
        },
    }


def build_ets_spec(error: str, trend: str | None, damped_trend: bool) -> dict[str, Any]:
    trend_label = trend if trend is not None else "none"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"ETS(error={error}, trend={trend_label}, damped_trend={damped_trend})",
        "ets": {
            "error": error,
            "trend": trend,
            "damped_trend": damped_trend,
        },
    }


def build_dols_fmols_spec(
    method: str,
    exogenous: list[str],
    trend: str,
    leads: int = 0,
    lags: int = 0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    dols_bits = f" leads={leads} lags={lags}" if method == "dols" else ""
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"{method.upper()} trend={trend}{dols_bits} with {label}",
        "dols_fmols": {
            "method": method,
            "exogenous": exogenous,
            "trend": trend,
            "leads": leads,
            "lags": lags,
        },
    }


def build_markov_switching_spec(exogenous: list[str]) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"Markov-Switching AR / ARX with {label}",
        "markov_switching": {
            "exogenous": exogenous,
        },
    }


def build_state_space_spec(
    level: str,
    cycle: bool,
    exogenous: list[str],
    damped_cycle: bool = False,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    cycle_label = "cycle" if cycle else "no_cycle"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"State-Space({level}, {cycle_label}) with {label}",
        "state_space": {
            "level": level,
            "cycle": cycle,
            "stochastic_cycle": cycle,
            "damped_cycle": damped_cycle,
            "exogenous": exogenous,
        },
    }


def generate_arimax_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    orders = [(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1)]
    for order in orders:
        for variable_set in unique_sets:
            candidates.append(build_arimax_spec(order, variable_set, trend="t"))

    return candidates


def generate_ardl_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for ar_lags in [1, 2, 4]:
            for distributed_lags in [1, 2, 4]:
                for trend in ["c", "ct"]:
                    candidates.append(
                        build_ardl_spec(
                            exogenous=exogenous,
                            ar_lags=ar_lags,
                            distributed_lags=distributed_lags,
                            trend=trend,
                        )
                    )
    return candidates


def generate_uecm_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=False,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for ar_lags in [1, 2, 4]:
            for distributed_lags in [1, 2, 4]:
                for trend in ["c", "ct"]:
                    candidates.append(
                        build_uecm_spec(
                            exogenous=exogenous,
                            ar_lags=ar_lags,
                            distributed_lags=distributed_lags,
                            trend=trend,
                        )
                    )
    return candidates


def generate_ets_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    del search_frame
    candidates: list[dict[str, Any]] = []
    for error in ["add"]:
        for trend, damped_trend in [(None, False), ("add", False), ("add", True)]:
            candidates.append(build_ets_spec(error=error, trend=trend, damped_trend=damped_trend))
    return candidates


def generate_dols_fmols_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=False,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for trend in ["c", "ct"]:
            candidates.append(build_dols_fmols_spec(method="fmols", exogenous=exogenous, trend=trend))
            for leads in [1, 2]:
                for lags in [1, 2]:
                    candidates.append(
                        build_dols_fmols_spec(
                            method="dols",
                            exogenous=exogenous,
                            trend=trend,
                            leads=leads,
                            lags=lags,
                        )
                    )
    return candidates


def generate_markov_switching_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=8,
        max_subset_size=2,
    )
    return [build_markov_switching_spec(exogenous=exogenous) for exogenous in unique_sets]


def generate_arimax_garch_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    mean_orders = [(1, 1, 0), (1, 1, 1)]
    variance_specs = [
        {"volatility": "GARCH", "p": 1, "o": 0, "q": 1, "distribution": "normal"},
        {"volatility": "GARCH", "p": 1, "o": 0, "q": 2, "distribution": "normal"},
        {"volatility": "GARCH", "p": 1, "o": 0, "q": 1, "distribution": "t"},
        {"volatility": "GARCH", "p": 2, "o": 0, "q": 1, "distribution": "normal"},
        {"volatility": "GARCH", "p": 2, "o": 0, "q": 2, "distribution": "normal"},
        {"volatility": "GARCH", "p": 1, "o": 0, "q": 2, "distribution": "t"},
        {"volatility": "GARCH", "p": 2, "o": 0, "q": 1, "distribution": "t"},
        {"volatility": "GARCH", "p": 2, "o": 0, "q": 2, "distribution": "t"},
    ]
    for variance_spec in variance_specs:
        for order in mean_orders:
            for variable_set in unique_sets:
                candidates.append(
                    build_arimax_garch_spec(
                        order=order,
                        exogenous=variable_set,
                        trend="t",
                        **variance_spec,
                    )
                )
    return candidates


def generate_state_space_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    structure_grid = [
        ("local level", False),
        ("local linear trend", False),
        ("local level", True),
        ("local linear trend", True),
    ]
    candidates: list[dict[str, Any]] = []
    for level, cycle in structure_grid:
        for variable_set in unique_sets:
            candidates.append(build_state_space_spec(level, cycle, variable_set))
    return candidates


def generate_state_space_refinement_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_sets = [
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "fed_funds"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "unemployment_rate", "building_permits"],
        ["mortgage_rate", "per_capita_income", "term_spread"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "unemployment_rate", "housing_inventory"],
    ]
    available_sets = []
    for variable_set in preferred_sets:
        filtered = [column for column in variable_set if column in search_frame.columns]
        if filtered:
            available_sets.append(filtered)

    candidates: list[dict[str, Any]] = []
    structure_grid = [
        ("local level", False),
        ("local linear trend", False),
        ("local level", True),
        ("local linear trend", True),
    ]
    for level, cycle in structure_grid:
        for variable_set in available_sets[:5]:
            candidates.append(build_state_space_spec(level, cycle, variable_set))
    return candidates


def generate_state_space_refinement2_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_sets = [
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income", "building_permits"],
        ["mortgage_rate", "unemployment_rate", "building_permits"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "term_spread"],
    ]
    candidates: list[dict[str, Any]] = []
    for variable_set in preferred_sets:
        filtered = [column for column in variable_set if column in search_frame.columns]
        if not filtered:
            continue
        candidates.append(build_state_space_spec("local level", False, filtered))
        candidates.append(build_state_space_spec("local linear trend", False, filtered))
        candidates.append(build_state_space_spec("local level", True, filtered))
        candidates.append(build_state_space_spec("local linear trend", True, filtered))
    return candidates


def build_var_spec(lags: int, endogenous: list[str], trend: str = "c") -> dict[str, Any]:
    label = ", ".join(endogenous[1:]) if len(endogenous) > 1 else "hpi only"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"VAR({lags}) trend={trend} with {label}",
        "var": {
            "lags": lags,
            "trend": trend,
            "endogenous": endogenous,
        },
    }


def build_bvar_spec(
    lags: int,
    endogenous: list[str],
    tightness: float,
    trend: str = "c",
    prior_type: str = "minnesota",
) -> dict[str, Any]:
    label = ", ".join(endogenous[1:]) if len(endogenous) > 1 else "hpi only"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"BVAR({lags}) prior={prior_type} lambda={tightness} trend={trend} with {label}",
        "bvar": {
            "lags": lags,
            "trend": trend,
            "prior_type": prior_type,
            "tightness": tightness,
            "lag_decay": 1.0,
            "endogenous": endogenous,
        },
    }


def build_threshold_var_spec(
    lags: int,
    endogenous: list[str],
    threshold_variable: str,
    delay: int,
    threshold_quantile: float,
    trend: str = "c",
    ridge_alpha: float = 0.1,
) -> dict[str, Any]:
    label = ", ".join(endogenous[1:]) if len(endogenous) > 1 else "hpi only"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"Threshold VAR({lags}) delay={delay} q={threshold_quantile} "
            f"threshold={threshold_variable} with {label}"
        ),
        "threshold_var": {
            "lags": lags,
            "trend": trend,
            "delay": delay,
            "threshold_variable": threshold_variable,
            "threshold_quantile": threshold_quantile,
            "ridge_alpha": ridge_alpha,
            "endogenous": endogenous,
            "regime_count": 2,
        },
    }


def build_vecm_spec(k_ar_diff: int, endogenous: list[str], deterministic: str = "ci") -> dict[str, Any]:
    label = ", ".join(endogenous[1:]) if len(endogenous) > 1 else "hpi only"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"VECM(k_ar_diff={k_ar_diff}) deterministic={deterministic} with {label}",
        "vecm": {
            "k_ar_diff": k_ar_diff,
            "deterministic": deterministic,
            "endogenous": endogenous,
        },
    }


def build_dynamic_factor_spec(
    factor_order: int,
    endogenous: list[str],
    k_factors: int = 1,
) -> dict[str, Any]:
    label = ", ".join(endogenous[1:]) if len(endogenous) > 1 else "hpi only"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": f"Dynamic Factor(k_factors={k_factors}, factor_order={factor_order}) with {label}",
        "dynamic_factor": {
            "k_factors": k_factors,
            "factor_order": factor_order,
            "error_order": 0,
            "error_var": False,
            "error_cov_type": "diagonal",
            "enforce_stationarity": True,
            "endogenous": endogenous,
        },
    }


def build_regularized_linear_spec(
    model_type: str,
    exogenous: list[str],
    target_lags: list[int],
    alpha: float,
    l1_ratio: float | None = None,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    family_label = {
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elastic_net": "Elastic Net",
    }[model_type]
    description = f"{family_label} alpha={alpha} lags={target_lags} with {label}"
    if model_type == "elastic_net" and l1_ratio is not None:
        description += f" l1_ratio={l1_ratio}"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": description,
        "regularized_linear": {
            "model_type": model_type,
            "exogenous": exogenous,
            "target_lags": target_lags,
            "alpha": alpha,
            "l1_ratio": 0.5 if l1_ratio is None else l1_ratio,
            "fit_intercept": True,
            "max_iter": 5000,
        },
    }


def build_random_forest_spec(
    exogenous: list[str],
    target_lags: list[int],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int = 1,
    max_features: float = 1.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    depth_label = "None" if max_depth is None else str(max_depth)
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"Random Forest n_estimators={n_estimators} max_depth={depth_label} "
            f"min_samples_leaf={min_samples_leaf} lags={target_lags} with {label}"
        ),
        "random_forest": {
            "exogenous": exogenous,
            "target_lags": target_lags,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        },
    }


def build_gradient_boosting_spec(
    exogenous: list[str],
    target_lags: list[int],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"Gradient Boosting n_estimators={n_estimators} learning_rate={learning_rate} "
            f"max_depth={max_depth} min_samples_leaf={min_samples_leaf} lags={target_lags} with {label}"
        ),
        "gradient_boosting": {
            "exogenous": exogenous,
            "target_lags": target_lags,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "subsample": subsample,
        },
    }


def build_support_vector_regression_spec(
    exogenous: list[str],
    target_lags: list[int],
    kernel: str,
    c: float,
    epsilon: float,
    gamma: str | float,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"Support Vector Regression kernel={kernel} C={c} epsilon={epsilon} "
            f"gamma={gamma} lags={target_lags} with {label}"
        ),
        "support_vector_regression": {
            "exogenous": exogenous,
            "target_lags": target_lags,
            "kernel": kernel,
            "c": c,
            "epsilon": epsilon,
            "gamma": gamma,
        },
    }


def build_xgboost_spec(
    exogenous: list[str],
    target_lags: list[int],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    min_child_weight: float,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"XGBoost n_estimators={n_estimators} learning_rate={learning_rate} max_depth={max_depth} "
            f"subsample={subsample} colsample_bytree={colsample_bytree} lags={target_lags} with {label}"
        ),
        "xgboost": {
            "exogenous": exogenous,
            "target_lags": target_lags,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
        },
    }


def build_lightgbm_spec(
    exogenous: list[str],
    target_lags: list[int],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    num_leaves: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    min_child_samples: int,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"LightGBM n_estimators={n_estimators} learning_rate={learning_rate} max_depth={max_depth} "
            f"num_leaves={num_leaves} lags={target_lags} with {label}"
        ),
        "lightgbm": {
            "exogenous": exogenous,
            "target_lags": target_lags,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "min_child_samples": min_child_samples,
        },
    }


def build_neuralprophet_spec(
    exogenous: list[str],
    n_lags: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    trend_reg: float = 0.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"NeuralProphet n_lags={n_lags} epochs={epochs} learning_rate={learning_rate} "
            f"batch_size={batch_size} with {label}"
        ),
        "neuralprophet": {
            "exogenous": exogenous,
            "n_lags": n_lags,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "trend_reg": trend_reg,
        },
    }


def build_lstm_gru_spec(
    cell_type: str,
    exogenous: list[str],
    lookback: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    dropout: float = 0.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    family_label = "LSTM" if cell_type == "lstm" else "GRU"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"{family_label} lookback={lookback} hidden_size={hidden_size} num_layers={num_layers} "
            f"epochs={epochs} learning_rate={learning_rate} with {label}"
        ),
        "lstm_gru": {
            "cell_type": cell_type,
            "exogenous": exogenous,
            "lookback": lookback,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
        },
    }


def build_tcn_spec(
    exogenous: list[str],
    lookback: int,
    channels: int,
    num_blocks: int,
    epochs: int,
    learning_rate: float,
    kernel_size: int = 2,
    dropout: float = 0.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"TCN lookback={lookback} channels={channels} num_blocks={num_blocks} "
            f"kernel_size={kernel_size} epochs={epochs} learning_rate={learning_rate} with {label}"
        ),
        "tcn": {
            "exogenous": exogenous,
            "lookback": lookback,
            "channels": channels,
            "num_blocks": num_blocks,
            "kernel_size": kernel_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
        },
    }


def build_nbeats_spec(
    exogenous: list[str],
    lookback: int,
    stack_width: int,
    n_blocks: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    dropout: float = 0.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"N-BEATS lookback={lookback} stack_width={stack_width} n_blocks={n_blocks} "
            f"n_layers={n_layers} epochs={epochs} learning_rate={learning_rate} with {label}"
        ),
        "nbeats": {
            "exogenous": exogenous,
            "lookback": lookback,
            "stack_width": stack_width,
            "n_blocks": n_blocks,
            "n_layers": n_layers,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
        },
    }


def build_transformer_spec(
    exogenous: list[str],
    lookback: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    feedforward_dim: int,
    epochs: int,
    learning_rate: float,
    dropout: float = 0.0,
) -> dict[str, Any]:
    label = ", ".join(exogenous) if exogenous else "no exogenous variables"
    return {
        "sample_start": DEFAULT_EXPERIMENT_SPEC["sample_start"],
        "target_column": DEFAULT_EXPERIMENT_SPEC["target_column"],
        "target_date_column": DEFAULT_EXPERIMENT_SPEC["target_date_column"],
        "champion_eligible": DEFAULT_EXPERIMENT_SPEC["champion_eligible"],
        "description": (
            f"Transformer lookback={lookback} model_dim={model_dim} num_heads={num_heads} "
            f"num_layers={num_layers} feedforward_dim={feedforward_dim} epochs={epochs} learning_rate={learning_rate} with {label}"
        ),
        "transformer": {
            "exogenous": exogenous,
            "lookback": lookback,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feedforward_dim": feedforward_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
        },
    }


def generate_var_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_endogenous = screened_system_subsets(
        search_frame,
        manifest,
        variable_config,
        max_ranked=10,
        max_subset_size=3,
    )
    candidates: list[dict[str, Any]] = []

    for endogenous in unique_endogenous:
        for lag_order in [1, 2, 4]:
            candidates.append(build_var_spec(lag_order, endogenous, trend="c"))
    return candidates


def generate_var_refinement_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    subsets = [
        ["term_spread"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "per_capita_income", "real_gdp"],
    ]
    available = set(search_frame.columns)
    candidates: list[dict[str, Any]] = []
    for subset in subsets:
        endogenous = ["hpi"] + [column for column in subset if column in available]
        if len(endogenous) < 2:
            continue
        for lag_order in [3, 4, 5, 6]:
            for trend in ["c", "ct"]:
                candidates.append(build_var_spec(lag_order, endogenous, trend=trend))
    return candidates


def generate_var_refinement2_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    subsets = [
        ["term_spread"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "term_spread", "real_gdp"],
        ["mortgage_rate", "per_capita_income", "real_gdp"],
    ]
    available = set(search_frame.columns)
    candidates: list[dict[str, Any]] = []
    for subset in subsets:
        endogenous = ["hpi"] + [column for column in subset if column in available]
        if len(endogenous) < 2:
            continue
        for lag_order in [4, 5, 6]:
            for trend in ["ct", "ctt"]:
                candidates.append(build_var_spec(lag_order, endogenous, trend=trend))
    return candidates


def generate_bvar_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_endogenous = screened_system_subsets(
        search_frame,
        manifest,
        variable_config,
        max_ranked=8,
        max_subset_size=3,
    )
    candidates: list[dict[str, Any]] = []

    lag_orders = [1, 2]
    tightness_values = [0.1, 0.2]
    for endogenous in unique_endogenous:
        for lag_order in lag_orders:
            for tightness in tightness_values:
                candidates.append(build_bvar_spec(lag_order, endogenous, tightness=tightness, trend="c"))

    for endogenous in unique_endogenous[:8]:
        for tightness in tightness_values:
            candidates.append(build_bvar_spec(4, endogenous, tightness=tightness, trend="c"))

    return candidates


def generate_bvar_refinement_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_sets = [
        ["building_permits"],
        ["per_capita_income"],
        ["real_gdp"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
    ]
    available = set(search_frame.columns)
    candidates: list[dict[str, Any]] = []
    for subset in preferred_sets:
        endogenous = ["hpi"] + [column for column in subset if column in available]
        if len(endogenous) < 2:
            continue
        for lag_order in [2, 3, 4]:
            for tightness in [0.15, 0.2, 0.25, 0.3]:
                candidates.append(build_bvar_spec(lag_order, endogenous, tightness=tightness, trend="c"))
    return candidates


def generate_bvar_refinement2_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_sets = [
        ["building_permits"],
        ["mortgage_rate", "building_permits"],
        ["building_permits", "term_spread"],
        ["mortgage_rate", "building_permits", "term_spread"],
        ["mortgage_rate", "real_gdp", "building_permits"],
    ]
    available = set(search_frame.columns)
    candidates: list[dict[str, Any]] = []
    for subset in preferred_sets:
        endogenous = ["hpi"] + [column for column in subset if column in available]
        if len(endogenous) < 2:
            continue
        for lag_order in [2, 3]:
            for tightness in [0.25, 0.3, 0.35, 0.4]:
                candidates.append(build_bvar_spec(lag_order, endogenous, tightness=tightness, trend="c"))
    return candidates


def generate_threshold_var_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_endogenous = screened_system_subsets(
        search_frame,
        manifest,
        variable_config,
        max_ranked=8,
        max_subset_size=3,
    )
    hyper_grid = [
        (1, 1, 0.4),
        (1, 1, 0.5),
        (1, 1, 0.6),
        (2, 1, 0.5),
        (1, 2, 0.5),
        (2, 2, 0.5),
    ]
    candidates: list[dict[str, Any]] = []
    for endogenous in unique_endogenous:
        threshold_variables = ["hpi"] + endogenous[1:]
        for threshold_variable in threshold_variables:
            for lags, delay, threshold_quantile in hyper_grid:
                candidates.append(
                    build_threshold_var_spec(
                        lags=lags,
                        endogenous=endogenous,
                        threshold_variable=threshold_variable,
                        delay=delay,
                        threshold_quantile=threshold_quantile,
                        trend="c",
                        ridge_alpha=0.1,
                    )
                )
    return candidates


def generate_threshold_var_refinement_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    candidate_shapes = [
        (["hpi", "mortgage_rate", "housing_inventory"], "housing_inventory"),
        (["hpi", "mortgage_rate", "real_gdp", "building_permits"], "building_permits"),
        (["hpi", "mortgage_rate", "building_permits"], "building_permits"),
        (["hpi", "mortgage_rate", "term_spread"], "term_spread"),
    ]
    available = set(search_frame.columns)
    hyper_grid = [
        (1, 1, 0.45),
        (1, 1, 0.5),
        (1, 1, 0.55),
        (1, 1, 0.6),
        (2, 2, 0.5),
    ]
    candidates: list[dict[str, Any]] = []
    for endogenous, threshold_variable in candidate_shapes:
        filtered = [column for column in endogenous if column == "hpi" or column in available]
        if len(filtered) < 2:
            continue
        if threshold_variable != "hpi" and threshold_variable not in filtered:
            continue
        for lags, delay, threshold_quantile in hyper_grid:
            candidates.append(
                build_threshold_var_spec(
                    lags=lags,
                    endogenous=filtered,
                    threshold_variable=threshold_variable,
                    delay=delay,
                    threshold_quantile=threshold_quantile,
                    trend="c",
                    ridge_alpha=0.05,
                )
            )
    return candidates


def generate_threshold_var_refinement2_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    candidate_shapes = [
        (["hpi", "mortgage_rate", "housing_inventory"], "housing_inventory"),
        (["hpi", "mortgage_rate", "housing_inventory", "building_permits"], "housing_inventory"),
        (["hpi", "mortgage_rate", "housing_inventory", "term_spread"], "housing_inventory"),
        (["hpi", "mortgage_rate", "real_gdp", "building_permits"], "building_permits"),
    ]
    available = set(search_frame.columns)
    hyper_grid = [
        (1, 1, 0.5, 0.025),
        (1, 1, 0.55, 0.025),
        (1, 1, 0.6, 0.025),
        (2, 1, 0.55, 0.025),
        (2, 2, 0.55, 0.05),
    ]
    candidates: list[dict[str, Any]] = []
    for endogenous, threshold_variable in candidate_shapes:
        filtered = [column for column in endogenous if column == "hpi" or column in available]
        if len(filtered) < 2:
            continue
        if threshold_variable != "hpi" and threshold_variable not in filtered:
            continue
        for lags, delay, threshold_quantile, ridge_alpha in hyper_grid:
            candidates.append(
                build_threshold_var_spec(
                    lags=lags,
                    endogenous=filtered,
                    threshold_variable=threshold_variable,
                    delay=delay,
                    threshold_quantile=threshold_quantile,
                    trend="c",
                    ridge_alpha=ridge_alpha,
                )
            )
    return candidates


def generate_dynamic_factor_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_endogenous = screened_system_subsets(
        search_frame,
        manifest,
        variable_config,
        max_ranked=8,
        max_subset_size=3,
    )
    candidates: list[dict[str, Any]] = []

    for endogenous in unique_endogenous:
        for factor_order in [1, 2, 4]:
            candidates.append(build_dynamic_factor_spec(factor_order, endogenous, k_factors=1))
    return candidates


def generate_vecm_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_endogenous = screened_system_subsets(
        search_frame,
        manifest,
        variable_config,
        max_ranked=8,
        max_subset_size=3,
    )
    candidates: list[dict[str, Any]] = []

    for endogenous in unique_endogenous:
        for k_ar_diff in [1, 2, 4]:
            candidates.append(build_vecm_spec(k_ar_diff, endogenous, deterministic="ci"))
    return candidates


def generate_vecm_refinement_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    subsets = [
        ["building_permits"],
        ["housing_starts"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "housing_starts", "building_permits"],
    ]
    available = set(search_frame.columns)
    candidates: list[dict[str, Any]] = []
    for subset in subsets:
        endogenous = ["hpi"] + [column for column in subset if column in available]
        if len(endogenous) < 2:
            continue
        for k_ar_diff in [1, 2, 3, 4]:
            for deterministic in ["ci", "co", "li"]:
                candidates.append(build_vecm_spec(k_ar_diff, endogenous, deterministic=deterministic))
    return candidates


def generate_vecm_refinement2_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    subsets = [
        ["building_permits"],
        ["mortgage_rate", "building_permits"],
        ["housing_starts", "building_permits"],
        ["mortgage_rate", "housing_starts", "building_permits"],
        ["mortgage_rate", "per_capita_income", "building_permits"],
    ]
    available = set(search_frame.columns)
    candidates: list[dict[str, Any]] = []
    for subset in subsets:
        endogenous = ["hpi"] + [column for column in subset if column in available]
        if len(endogenous) < 2:
            continue
        for k_ar_diff in [1, 2]:
            for deterministic in ["co", "ci", "li", "lo"]:
                candidates.append(build_vecm_spec(k_ar_diff, endogenous, deterministic=deterministic))
    return candidates


def generate_regularized_linear_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    lag_sets = [[1, 2, 4], [1, 4, 8], [1, 2, 4, 8]]
    for exogenous in unique_sets:
        for target_lags in lag_sets:
            for alpha in [0.01, 0.1, 1.0]:
                candidates.append(build_regularized_linear_spec("ridge", exogenous, target_lags, alpha=alpha))
                candidates.append(build_regularized_linear_spec("lasso", exogenous, target_lags, alpha=alpha))
            for alpha in [0.01, 0.1, 1.0]:
                for l1_ratio in [0.25, 0.5, 0.75]:
                    candidates.append(
                        build_regularized_linear_spec(
                            "elastic_net",
                            exogenous,
                            target_lags,
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                        )
                    )
    return candidates


def generate_random_forest_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    lag_sets = [[1, 2, 4], [1, 4, 8], [1, 2, 4, 8]]
    for exogenous in unique_sets:
        for target_lags in lag_sets:
            for n_estimators in [200, 500]:
                for max_depth in [3, 5, None]:
                    for min_samples_leaf in [1, 4]:
                        candidates.append(
                            build_random_forest_spec(
                                exogenous=exogenous,
                                target_lags=target_lags,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                max_features=1.0,
                            )
                        )
    return candidates


def generate_gradient_boosting_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    lag_sets = [[1, 2, 4], [1, 4, 8], [1, 2, 4, 8]]
    for exogenous in unique_sets:
        for target_lags in lag_sets:
            for n_estimators in [200, 500]:
                for learning_rate in [0.03, 0.05, 0.1]:
                    for max_depth in [2, 3]:
                        for min_samples_leaf in [1, 4]:
                            candidates.append(
                                build_gradient_boosting_spec(
                                    exogenous=exogenous,
                                    target_lags=target_lags,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    subsample=1.0,
                                )
                            )
    return candidates


def generate_support_vector_regression_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    lag_sets = [[1, 2, 4], [1, 4, 8], [1, 2, 4, 8]]
    kernels = [
        ("linear", "scale"),
        ("rbf", "scale"),
        ("rbf", 0.1),
    ]
    for exogenous in unique_sets:
        for target_lags in lag_sets:
            for kernel, gamma in kernels:
                for c in [1.0, 10.0, 50.0]:
                    for epsilon in [0.1, 1.0, 5.0]:
                        candidates.append(
                            build_support_vector_regression_spec(
                                exogenous=exogenous,
                                target_lags=target_lags,
                                kernel=kernel,
                                c=c,
                                epsilon=epsilon,
                                gamma=gamma,
                            )
                        )
    return candidates


def generate_xgboost_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    lag_sets = [[1, 2, 4], [1, 4, 8], [1, 2, 4, 8]]
    for exogenous in unique_sets:
        for target_lags in lag_sets:
            for n_estimators in [200, 500]:
                for learning_rate in [0.03, 0.05, 0.1]:
                    for max_depth in [2, 3, 4]:
                        candidates.append(
                            build_xgboost_spec(
                                exogenous=exogenous,
                                target_lags=target_lags,
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_lambda=1.0,
                                min_child_weight=1.0,
                            )
                        )
    return candidates


def generate_lightgbm_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=10,
        max_subset_size=3,
    )

    candidates: list[dict[str, Any]] = []
    lag_sets = [[1, 2, 4], [1, 4, 8], [1, 2, 4, 8]]
    for exogenous in unique_sets:
        for target_lags in lag_sets:
            for n_estimators in [200, 500]:
                for learning_rate in [0.03, 0.05, 0.1]:
                    for max_depth, num_leaves in [(2, 7), (3, 15), (4, 31)]:
                        candidates.append(
                            build_lightgbm_spec(
                                exogenous=exogenous,
                                target_lags=target_lags,
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                num_leaves=num_leaves,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_lambda=0.0,
                                min_child_samples=10,
                            )
                        )
    return candidates


def generate_neuralprophet_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=8,
        max_subset_size=2,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for n_lags in [4, 8, 12]:
            for epochs in [50, 100]:
                for learning_rate in [0.01, 0.03]:
                    candidates.append(
                        build_neuralprophet_spec(
                            exogenous=exogenous,
                            n_lags=n_lags,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            batch_size=16,
                            trend_reg=0.0,
                        )
                    )
    return candidates


def generate_lstm_gru_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=8,
        max_subset_size=2,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for cell_type in ["lstm", "gru"]:
            for lookback in [4, 8]:
                for hidden_size in [16, 32]:
                    for num_layers in [1, 2]:
                        candidates.append(
                            build_lstm_gru_spec(
                                cell_type=cell_type,
                                exogenous=exogenous,
                                lookback=lookback,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                epochs=75,
                                learning_rate=0.01,
                                dropout=0.0,
                            )
                        )
    return candidates


def generate_tcn_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=8,
        max_subset_size=2,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for lookback in [4, 8]:
            for channels in [16, 32]:
                for num_blocks in [2, 3]:
                    candidates.append(
                        build_tcn_spec(
                            exogenous=exogenous,
                            lookback=lookback,
                            channels=channels,
                            num_blocks=num_blocks,
                            kernel_size=2,
                            epochs=75,
                            learning_rate=0.01,
                            dropout=0.0,
                        )
                    )
    return candidates


def generate_nbeats_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=8,
        max_subset_size=2,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for lookback in [4, 8]:
            for stack_width in [32, 64]:
                for n_blocks in [2, 3]:
                    candidates.append(
                        build_nbeats_spec(
                            exogenous=exogenous,
                            lookback=lookback,
                            stack_width=stack_width,
                            n_blocks=n_blocks,
                            n_layers=2,
                            epochs=75,
                            learning_rate=0.01,
                            dropout=0.0,
                        )
                    )
    return candidates


def generate_transformer_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, manifest, variable_config = load_panels()
    unique_sets = screened_exogenous_subsets(
        search_frame,
        manifest,
        variable_config,
        include_empty=True,
        max_ranked=8,
        max_subset_size=2,
    )

    candidates: list[dict[str, Any]] = []
    for exogenous in unique_sets:
        for lookback in [4, 8]:
            for model_dim, num_heads in [(16, 2), (32, 4)]:
                for num_layers in [1, 2]:
                    candidates.append(
                        build_transformer_spec(
                            exogenous=exogenous,
                            lookback=lookback,
                            model_dim=model_dim,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            feedforward_dim=model_dim * 2,
                            epochs=75,
                            learning_rate=0.01,
                            dropout=0.0,
                        )
                    )
    return candidates


def run_single_trial(args: argparse.Namespace) -> int:
    if not args.trial_id or not args.output_dir:
        raise RuntimeError("trial-id and output-dir are required for search/finalize modes.")

    output_dir = Path(args.output_dir).resolve()
    ensure_parent_dirs(output_dir)
    logger = TrialLogger(output_dir)
    logger.info(f"Starting {args.mode} for {args.model_class} trial {args.trial_id}.")
    write_snapshot(output_dir)
    write_spec(output_dir, args, resolve_dependency_scope(args.model_class))

    try:
        if args.mode == "search":
            metrics = execute_search(args, output_dir, logger)
        else:
            metrics = execute_finalize(args, output_dir, logger)
        write_metrics(output_dir, metrics)
        persist_logs(metrics, output_dir)
        return 0
    except Exception as exc:  # pragma: no cover
        error_summary = str(exc)
        logger.error(error_summary)
        logger.error(traceback.format_exc())
        metrics = {
            "experiment_id": args.experiment_id,
            "trial_id": args.trial_id,
            "model_class": args.model_class,
            "champion_eligible": bool(get_active_spec()["champion_eligible"]),
            "description": get_active_spec()["description"],
            "gof_composite": np.nan,
            "gof_insample": np.nan,
            "gof_validation_near": np.nan,
            "gof_validation_far": np.nan,
            "gof_diagnostic": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "theil_u": np.nan,
            "directional_accuracy": np.nan,
            "n_params": 0,
            "status": "error",
            "diagnostics_passed": [],
            "diagnostics_failed": [],
            "error_summary": error_summary,
            "rmse_1yr": np.nan,
            "rmse_3yr": np.nan,
        }
        write_metrics(output_dir, metrics)
        persist_logs(metrics, output_dir)
        return 1


def run_experiment(args: argparse.Namespace) -> int:
    search_frame, _, _, _ = load_panels()
    search_frame = filter_sample(search_frame)
    experiment_dir = ROOT / "experiments" / args.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    refinement = refinement_stage(args.experiment_id)

    if args.model_class == "ARIMAX":
        candidates = generate_arimax_candidate_specs(search_frame)
    elif args.model_class == "ARDL":
        candidates = generate_ardl_candidate_specs(search_frame)
    elif args.model_class == "UECM / ECM":
        candidates = generate_uecm_candidate_specs(search_frame)
    elif args.model_class == "DOLS / FMOLS":
        candidates = generate_dols_fmols_candidate_specs(search_frame)
    elif args.model_class == "ETS":
        candidates = generate_ets_candidate_specs(search_frame)
    elif args.model_class == "Markov-Switching AR / ARX":
        candidates = generate_markov_switching_candidate_specs(search_frame)
    elif args.model_class == "ARIMAX-GARCH":
        candidates = generate_arimax_garch_candidate_specs(search_frame)
    elif args.model_class == "State-Space":
        if refinement == 2:
            candidates = generate_state_space_refinement2_candidate_specs(search_frame)
        elif refinement == 1:
            candidates = generate_state_space_refinement_candidate_specs(search_frame)
        else:
            candidates = generate_state_space_candidate_specs(search_frame)
    elif args.model_class == "Dynamic Factor":
        candidates = generate_dynamic_factor_candidate_specs(search_frame)
    elif args.model_class == "Ridge / Lasso / Elastic Net":
        candidates = generate_regularized_linear_candidate_specs(search_frame)
    elif args.model_class == "Random Forest":
        candidates = generate_random_forest_candidate_specs(search_frame)
    elif args.model_class == "Gradient Boosting":
        candidates = generate_gradient_boosting_candidate_specs(search_frame)
    elif args.model_class == "Support Vector Regression":
        candidates = generate_support_vector_regression_candidate_specs(search_frame)
    elif args.model_class == "XGBoost":
        candidates = generate_xgboost_candidate_specs(search_frame)
    elif args.model_class == "LightGBM":
        candidates = generate_lightgbm_candidate_specs(search_frame)
    elif args.model_class == "NeuralProphet":
        candidates = generate_neuralprophet_candidate_specs(search_frame)
    elif args.model_class == "LSTM / GRU":
        candidates = generate_lstm_gru_candidate_specs(search_frame)
    elif args.model_class == "TCN":
        candidates = generate_tcn_candidate_specs(search_frame)
    elif args.model_class == "N-BEATS":
        candidates = generate_nbeats_candidate_specs(search_frame)
    elif args.model_class == "Transformer":
        candidates = generate_transformer_candidate_specs(search_frame)
    elif args.model_class == "VAR":
        if refinement == 2:
            candidates = generate_var_refinement2_candidate_specs(search_frame)
        elif refinement == 1:
            candidates = generate_var_refinement_candidate_specs(search_frame)
        else:
            candidates = generate_var_candidate_specs(search_frame)
    elif args.model_class == "BVAR":
        if refinement == 2:
            candidates = generate_bvar_refinement2_candidate_specs(search_frame)
        elif refinement == 1:
            candidates = generate_bvar_refinement_candidate_specs(search_frame)
        else:
            candidates = generate_bvar_candidate_specs(search_frame)
    elif args.model_class in {"Threshold VAR", "Threshold VAR / SETAR"}:
        if refinement == 2:
            candidates = generate_threshold_var_refinement2_candidate_specs(search_frame)
        elif refinement == 1:
            candidates = generate_threshold_var_refinement_candidate_specs(search_frame)
        else:
            candidates = generate_threshold_var_candidate_specs(search_frame)
    elif args.model_class == "VECM":
        if refinement == 2:
            candidates = generate_vecm_refinement2_candidate_specs(search_frame)
        elif refinement == 1:
            candidates = generate_vecm_refinement_candidate_specs(search_frame)
        else:
            candidates = generate_vecm_candidate_specs(search_frame)
    else:
        raise NotImplementedError(f"Experiment mode is not implemented for {args.model_class}.")
    candidate_limit = candidate_cap_for_experiment(args.model_class, refinement, args.max_trials)
    candidates = candidates[:candidate_limit]
    start_offset = existing_trial_count(experiment_dir)
    if start_offset >= len(candidates):
        print(f"No remaining {args.model_class} candidate specs to run for this experiment.")
        return 0

    failures = 0
    completed = 0
    stagnant_successes = 0
    best_gof = -np.inf
    results_path = experiment_dir / "results.tsv"
    if results_path.exists():
        existing_results = pd.read_csv(results_path, sep="\t")
        existing_results["gof_composite"] = pd.to_numeric(existing_results["gof_composite"], errors="coerce")
        ok_results = existing_results.loc[existing_results["status"] == "ok", "gof_composite"].dropna()
        if not ok_results.empty:
            best_gof = float(ok_results.max())

    for index, candidate in enumerate(candidates[start_offset:], start=start_offset + 1):
        if completed >= args.max_trials:
            print(f"Reached max trials limit of {args.max_trials} for experiment {args.experiment_id}.")
            break
        set_active_spec(candidate)
        trial_id = f"{args.experiment_id}_{index:03d}"
        output_dir = experiment_dir / "runs" / trial_id
        trial_args = argparse.Namespace(
            mode="search",
            model_class=args.model_class,
            experiment_id=args.experiment_id,
            trial_id=trial_id,
            output_dir=str(output_dir),
            max_trials=args.max_trials,
            improvement_threshold=args.improvement_threshold,
            patience=args.patience,
        )
        exit_code = run_single_trial(trial_args)
        completed += 1
        if exit_code == 0:
            failures = 0
            metrics_path = output_dir / "metrics.json"
            trial_gof = np.nan
            if metrics_path.exists():
                trial_metrics = load_json(metrics_path)
                trial_gof = float(pd.to_numeric(trial_metrics.get("gof_composite"), errors="coerce"))
            if np.isfinite(trial_gof):
                if trial_gof >= best_gof + float(args.improvement_threshold):
                    best_gof = float(trial_gof)
                    stagnant_successes = 0
                else:
                    stagnant_successes += 1
            else:
                stagnant_successes += 1
        else:
            failures += 1
        if failures >= 5:
            print("Stopping experiment after 5 consecutive failed trials.")
            break
        if stagnant_successes >= args.patience:
            print(
                "Stopping experiment after "
                f"{args.patience} successful trials without GOF improvement "
                f">= {args.improvement_threshold:.4f}."
            )
            break

    print(f"Completed {completed} trial(s) for experiment {args.experiment_id}.")
    return 0


def main() -> int:
    args = parse_args()
    set_active_spec(DEFAULT_EXPERIMENT_SPEC)
    if args.mode == "experiment":
        return run_experiment(args)
    return run_single_trial(args)


if __name__ == "__main__":
    sys.exit(main())
