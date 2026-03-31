#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


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
    parser.add_argument("--max-trials", type=int, default=5)
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


def in_sample_metrics(frame: pd.DataFrame) -> tuple[Any, float, float, float]:
    exog_columns = get_active_spec()["arimax"]["exogenous"]
    fitted = fit_arimax(frame["hpi"], build_exog(frame, exog_columns))
    predicted = pd.Series(np.asarray(fitted.fittedvalues), index=frame.index)
    aligned = pd.DataFrame({"y": frame["hpi"], "pred": predicted}).dropna()
    insample_rmse = rmse(aligned["y"], aligned["pred"])
    naive_rmse = compute_naive_insample_rmse(frame["hpi"])
    gof = 1.0 - (insample_rmse / naive_rmse if naive_rmse else np.inf)
    return fitted, gof, insample_rmse, naive_rmse


def diagnostic_score(
    fitted,
    residuals: pd.Series,
    future_forecast: pd.Series,
    exog_columns: list[str],
    variable_config: dict[str, Any],
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
        if float(breaks_cusumolsresid(resids, ddof=len(fitted.params))[1]) > 0.05:
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
            actual = float(fitted.params.get(variable, 0.0))
            if expected == "positive" and actual <= 0:
                sign_fail = True
            if expected == "negative" and actual >= 0:
                sign_fail = True
        if sign_fail:
            fails.append("economic_sign")
        else:
            passes.append("economic_sign")

    changes = future_forecast.pct_change().dropna()
    cumulative = (future_forecast.iloc[-1] / future_forecast.iloc[0]) - 1.0 if len(future_forecast) > 1 else 0.0
    plausibility_ok = (
        (future_forecast > 0).all()
        and (changes.abs() < 0.15).all()
        and (-0.20 <= cumulative <= 5.0)
        and (float(future_forecast.std()) > 0.0)
    )
    if plausibility_ok:
        passes.append("forecast_plausibility")
    else:
        fails.append("forecast_plausibility")

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
    board = board.sort_values("gof_composite", ascending=False).drop_duplicates("experiment_id", keep="first").reset_index(drop=True)
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
    if args.model_class != "ARIMAX":
        raise NotImplementedError(
            f"{args.model_class} is not implemented yet. MVP support currently covers ARIMAX only."
        )

    search_frame, _, manifest, variable_config = load_panels()
    search_frame = filter_sample(search_frame)
    logger.info(f"Loaded {len(search_frame)} search rows for {args.model_class}.")

    fitted, gof_insample, insample_rmse, naive_insample_rmse = in_sample_metrics(search_frame)
    validation_frame, gof_near, gof_far, rmse_1yr, rmse_3yr, naive_rmse_1yr = walk_forward_validation(search_frame, manifest)
    validation_frame["experiment_id"] = args.experiment_id
    validation_frame["trial_id"] = args.trial_id
    validation_frame.to_parquet(output_dir / "validation_predictions.parquet", index=False)

    future_forecast = build_final_forecast(search_frame, variable_config, output_dir)
    gof_diag, passed, failed = diagnostic_score(
        fitted,
        pd.Series(np.asarray(fitted.resid)),
        future_forecast["hpi_forecast"],
        get_active_spec()["arimax"]["exogenous"],
        variable_config,
    )

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
        "n_params": int(len(fitted.params)),
        "status": "ok",
        "diagnostics_passed": passed,
        "diagnostics_failed": failed,
        "error_summary": "",
        "rmse_1yr": float(rmse_1yr),
        "rmse_3yr": float(rmse_3yr),
        "insample_rmse": float(insample_rmse),
        "naive_insample_rmse": float(naive_insample_rmse),
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
    for variable_set in unique_sets:
        for order in orders:
            candidates.append(build_arimax_spec(order, variable_set, trend="t"))

    baseline_no_exog = build_arimax_spec((1, 1, 0), [], trend="t")
    return [baseline_no_exog] + candidates


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
    if args.model_class != "ARIMAX":
        raise NotImplementedError("Experiment mode is currently implemented for ARIMAX only.")

    search_frame, _, _, _ = load_panels()
    search_frame = filter_sample(search_frame)
    experiment_dir = ROOT / "experiments" / args.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_arimax_candidate_specs(search_frame)
    start_offset = existing_trial_count(experiment_dir)
    if start_offset >= len(candidates):
        print("No remaining ARIMAX candidate specs to run for this experiment.")
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
