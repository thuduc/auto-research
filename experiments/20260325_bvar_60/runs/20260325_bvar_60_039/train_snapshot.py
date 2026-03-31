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
from scipy.stats import jarque_bera, norm, t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
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
    parser.add_argument("--max-trials", type=int, default=60)
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
    if model_class in {"ARIMAX", "VAR", "VECM", "State-Space", "Dynamic Factor", "BVAR", "Threshold VAR", "Threshold VAR / SETAR"}:
        return "econometric"
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


def in_sample_metrics(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    exog_columns = get_active_spec()["arimax"]["exogenous"]
    fitted = fit_arimax(frame["hpi"], build_exog(frame, exog_columns))
    predicted = pd.Series(np.asarray(fitted.fittedvalues), index=frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
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
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
        "consumer_confidence",
        "fed_funds",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]
    if "mortgage_rate" not in available:
        raise RuntimeError("mortgage_rate is required for the ARIMAX MVP experiment.")

    variable_sets: list[list[str]] = [["mortgage_rate"]]
    for column in available:
        if column != "mortgage_rate":
            variable_sets.append(["mortgage_rate", column])
    triples = [
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "housing_starts", "term_spread"],
        ["mortgage_rate", "consumer_confidence", "housing_inventory"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "unemployment_rate", "real_gdp"],
        ["mortgage_rate", "unemployment_rate", "consumer_confidence"],
        ["mortgage_rate", "per_capita_income", "real_gdp"],
        ["mortgage_rate", "per_capita_income", "housing_starts"],
    ]
    for variable_set in triples:
        candidate = [column for column in variable_set if column in available]
        if len(candidate) >= 2:
            variable_sets.append(candidate)

    unique_sets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variable_set in variable_sets:
        deduped = tuple(dict.fromkeys(variable_set))
        if deduped not in seen:
            seen.add(deduped)
            unique_sets.append(list(deduped))

    candidates: list[dict[str, Any]] = []
    orders = [(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1)]
    for order in orders:
        for variable_set in unique_sets:
            candidates.append(build_arimax_spec(order, variable_set, trend="t"))

    baseline_no_exog = build_arimax_spec((1, 1, 0), [], trend="t")
    return [baseline_no_exog] + candidates


def generate_arimax_garch_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
        "consumer_confidence",
        "fed_funds",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]

    variable_sets: list[list[str]] = [
        [],
        ["mortgage_rate"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "housing_starts"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "consumer_confidence"],
        ["mortgage_rate", "fed_funds"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "housing_starts", "term_spread"],
        ["mortgage_rate", "consumer_confidence", "housing_inventory"],
    ]

    unique_sets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variable_set in variable_sets:
        filtered = [column for column in variable_set if column in available]
        deduped = tuple(dict.fromkeys(filtered))
        if deduped not in seen:
            seen.add(deduped)
            unique_sets.append(list(deduped))

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
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
        "consumer_confidence",
        "fed_funds",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]
    variable_sets: list[list[str]] = [
        [],
        ["mortgage_rate"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "housing_starts"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "consumer_confidence"],
        ["mortgage_rate", "fed_funds"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "housing_starts", "term_spread"],
        ["mortgage_rate", "consumer_confidence", "housing_inventory"],
        ["mortgage_rate", "real_gdp", "building_permits"],
    ]

    unique_sets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variable_set in variable_sets:
        filtered = [column for column in variable_set if column in available]
        deduped = tuple(dict.fromkeys(filtered))
        if deduped not in seen:
            seen.add(deduped)
            unique_sets.append(list(deduped))

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


def generate_var_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]
    candidates: list[dict[str, Any]] = []
    subsets = [
        ["mortgage_rate"],
        ["unemployment_rate"],
        ["per_capita_income"],
        ["real_gdp"],
        ["housing_starts"],
        ["building_permits"],
        ["term_spread"],
        ["housing_inventory"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "housing_starts"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "housing_starts", "building_permits"],
        ["mortgage_rate", "unemployment_rate", "housing_inventory"],
        ["mortgage_rate", "per_capita_income", "real_gdp"],
    ]
    unique_endogenous: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for subset in subsets:
        retained = ["hpi"] + [column for column in subset if column in available]
        if len(retained) < 2:
            continue
        key = tuple(retained)
        if key not in seen:
            seen.add(key)
            unique_endogenous.append(retained)

    for endogenous in unique_endogenous:
        for lag_order in [1, 2, 4]:
            candidates.append(build_var_spec(lag_order, endogenous, trend="c"))
    return candidates


def generate_bvar_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]
    candidates: list[dict[str, Any]] = []
    subsets = [
        ["mortgage_rate"],
        ["unemployment_rate"],
        ["per_capita_income"],
        ["real_gdp"],
        ["building_permits"],
        ["housing_inventory"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "housing_starts", "term_spread"],
    ]
    unique_endogenous: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for subset in subsets:
        retained = ["hpi"] + [column for column in subset if column in available]
        if len(retained) < 2:
            continue
        key = tuple(retained)
        if key not in seen:
            seen.add(key)
            unique_endogenous.append(retained)

    lag_orders = [1, 2]
    tightness_values = [0.1, 0.2]
    for endogenous in unique_endogenous:
        for lag_order in lag_orders:
            for tightness in tightness_values:
                candidates.append(build_bvar_spec(lag_order, endogenous, tightness=tightness, trend="c"))

    deeper_specs = [
        (["hpi", "mortgage_rate", "unemployment_rate", "per_capita_income"], 4, 0.1),
        (["hpi", "mortgage_rate", "real_gdp", "building_permits"], 4, 0.1),
        (["hpi", "mortgage_rate", "term_spread"], 4, 0.1),
        (["hpi", "mortgage_rate", "housing_inventory"], 4, 0.1),
        (["hpi", "mortgage_rate", "unemployment_rate", "per_capita_income"], 4, 0.2),
        (["hpi", "mortgage_rate", "real_gdp", "building_permits"], 4, 0.2),
        (["hpi", "mortgage_rate", "term_spread"], 4, 0.2),
        (["hpi", "mortgage_rate", "housing_inventory"], 4, 0.2),
    ]
    for endogenous, lag_order, tightness in deeper_specs:
        filtered = [column for column in endogenous if column == "hpi" or column in available]
        if len(filtered) >= 2:
            candidates.append(build_bvar_spec(lag_order, filtered, tightness=tightness, trend="c"))

    return candidates[:60]


def generate_dynamic_factor_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]
    candidates: list[dict[str, Any]] = []
    subsets = [
        ["mortgage_rate"],
        ["unemployment_rate"],
        ["per_capita_income"],
        ["real_gdp"],
        ["housing_starts"],
        ["building_permits"],
        ["term_spread"],
        ["housing_inventory"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "housing_starts"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "housing_starts", "building_permits"],
        ["mortgage_rate", "unemployment_rate", "housing_inventory"],
        ["mortgage_rate", "per_capita_income", "real_gdp"],
    ]
    unique_endogenous: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for subset in subsets:
        retained = ["hpi"] + [column for column in subset if column in available]
        if len(retained) < 2:
            continue
        key = tuple(retained)
        if key not in seen:
            seen.add(key)
            unique_endogenous.append(retained)

    for endogenous in unique_endogenous:
        for factor_order in [1, 2, 4]:
            candidates.append(build_dynamic_factor_spec(factor_order, endogenous, k_factors=1))
    return candidates


def generate_vecm_candidate_specs(search_frame: pd.DataFrame) -> list[dict[str, Any]]:
    preferred_variables = [
        "mortgage_rate",
        "unemployment_rate",
        "per_capita_income",
        "real_gdp",
        "housing_starts",
        "building_permits",
        "term_spread",
        "housing_inventory",
    ]
    available = [column for column in preferred_variables if column in search_frame.columns]
    candidates: list[dict[str, Any]] = []
    subsets = [
        ["mortgage_rate"],
        ["unemployment_rate"],
        ["per_capita_income"],
        ["real_gdp"],
        ["housing_starts"],
        ["building_permits"],
        ["term_spread"],
        ["housing_inventory"],
        ["mortgage_rate", "unemployment_rate"],
        ["mortgage_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp"],
        ["mortgage_rate", "housing_starts"],
        ["mortgage_rate", "building_permits"],
        ["mortgage_rate", "term_spread"],
        ["mortgage_rate", "housing_inventory"],
        ["mortgage_rate", "unemployment_rate", "per_capita_income"],
        ["mortgage_rate", "real_gdp", "building_permits"],
        ["mortgage_rate", "housing_starts", "building_permits"],
        ["mortgage_rate", "unemployment_rate", "housing_inventory"],
        ["mortgage_rate", "per_capita_income", "real_gdp"],
    ]
    unique_endogenous: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for subset in subsets:
        retained = ["hpi"] + [column for column in subset if column in available]
        if len(retained) < 2:
            continue
        key = tuple(retained)
        if key not in seen:
            seen.add(key)
            unique_endogenous.append(retained)

    for endogenous in unique_endogenous:
        for k_ar_diff in [1, 2, 4]:
            candidates.append(build_vecm_spec(k_ar_diff, endogenous, deterministic="ci"))
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

    if args.model_class == "ARIMAX":
        candidates = generate_arimax_candidate_specs(search_frame)
    elif args.model_class == "ARIMAX-GARCH":
        candidates = generate_arimax_garch_candidate_specs(search_frame)
    elif args.model_class == "State-Space":
        candidates = generate_state_space_candidate_specs(search_frame)
    elif args.model_class == "Dynamic Factor":
        candidates = generate_dynamic_factor_candidate_specs(search_frame)
    elif args.model_class == "VAR":
        candidates = generate_var_candidate_specs(search_frame)
    elif args.model_class == "BVAR":
        candidates = generate_bvar_candidate_specs(search_frame)
    elif args.model_class == "VECM":
        candidates = generate_vecm_candidate_specs(search_frame)
    else:
        raise NotImplementedError(f"Experiment mode is not implemented for {args.model_class}.")
    start_offset = existing_trial_count(experiment_dir)
    if start_offset >= len(candidates):
        print(f"No remaining {args.model_class} candidate specs to run for this experiment.")
        return 0

    planned = candidates[start_offset : start_offset + args.max_trials]
    failures = 0
    completed = 0
    for index, candidate in enumerate(planned, start=start_offset + 1):
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
        )
        exit_code = run_single_trial(trial_args)
        completed += 1
        if exit_code == 0:
            failures = 0
        else:
            failures += 1
        if failures >= 5:
            print("Stopping experiment after 5 consecutive failed trials.")
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
