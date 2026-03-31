#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train


ROOT = Path(__file__).resolve().parent
DEFAULT_COMPONENT_CLASSES = ["Gradient Boosting", "XGBoost", "N-BEATS", "Transformer"]
METHODS = ["simple_average", "inverse_rmse", "ridge_stacking"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--component-classes",
        nargs="+",
        default=DEFAULT_COMPONENT_CLASSES,
    )
    return parser.parse_args()


def ensure_experiment_dirs(experiment_id: str) -> Path:
    experiment_dir = ROOT / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def load_leaderboard() -> pd.DataFrame:
    board = pd.read_csv(ROOT / "leaderboard.tsv", sep="\t")
    board["gof_composite"] = pd.to_numeric(board["gof_composite"], errors="coerce")
    return board.sort_values("gof_composite", ascending=False).reset_index(drop=True)


def select_component_rows(model_classes: list[str]) -> list[pd.Series]:
    board = load_leaderboard()
    rows: list[pd.Series] = []
    for model_class in model_classes:
        subset = board.loc[board["model_class"] == model_class]
        if subset.empty:
            raise RuntimeError(f"No leaderboard entry found for model class {model_class}.")
        rows.append(subset.iloc[0])
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def component_artifacts(row: pd.Series) -> dict[str, Any]:
    experiment_id = str(row["experiment_id"])
    trial_id = str(row["best_trial_id"])
    run_dir = ROOT / "experiments" / experiment_id / "runs" / trial_id
    metrics = read_json(run_dir / "metrics.json")
    return {
        "component_name": str(row["model_class"]),
        "experiment_id": experiment_id,
        "trial_id": trial_id,
        "run_dir": run_dir,
        "metrics": metrics,
        "leaderboard_row": row.to_dict(),
    }


def load_component_validation(component: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(component["run_dir"] / "validation_predictions.parquet")
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    key = component["component_name"]
    return frame[["origin_date", "forecast_date", "horizon_q", "y_true", "y_pred"]].rename(columns={"y_pred": key})


def merge_validation_frames(components: list[dict[str, Any]]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component in components:
        frame = load_component_validation(component)
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(
                frame,
                on=["origin_date", "forecast_date", "horizon_q", "y_true"],
                how="inner",
            )
    if merged is None or merged.empty:
        raise RuntimeError("No validation predictions were available for ensemble construction.")
    return merged.sort_values(["origin_date", "forecast_date"]).reset_index(drop=True)


def build_naive_lookup() -> dict[tuple[pd.Timestamp, pd.Timestamp], float]:
    search_frame, _, manifest, _ = train.load_panels()
    search_frame = train.filter_sample(search_frame)
    origins = [pd.Timestamp(item) for item in manifest["backtest_origins"]]
    lookup: dict[tuple[pd.Timestamp, pd.Timestamp], float] = {}
    for origin in origins:
        origin_idx = search_frame.index[search_frame["date"] == origin]
        if len(origin_idx) != 1:
            continue
        end_idx = int(origin_idx[0])
        train_frame = search_frame.iloc[: end_idx + 1].copy()
        future = search_frame.iloc[end_idx + 1 : end_idx + 41].copy()
        if len(future) < 40:
            continue
        naive = train.forecast_naive_path(train_frame["hpi"], len(future))
        for forecast_date, value in zip(pd.to_datetime(future["date"]), naive):
            lookup[(origin, forecast_date)] = float(value)
    return lookup


def attach_naive_predictions(validation: pd.DataFrame) -> pd.DataFrame:
    lookup = build_naive_lookup()
    naive_values = [
        lookup[(pd.Timestamp(origin), pd.Timestamp(forecast))]
        for origin, forecast in zip(validation["origin_date"], validation["forecast_date"])
    ]
    enriched = validation.copy()
    enriched["naive_pred"] = naive_values
    return enriched


def validation_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    near = frame.loc[frame["horizon_q"] <= 12].copy()
    far = frame.loc[frame["horizon_q"] > 12].copy()

    rmse_1yr = train.rmse(near["y_true"], near[prediction_column])
    rmse_3yr = train.rmse(far["y_true"], far[prediction_column])
    naive_rmse_1yr = train.rmse(near["y_true"], near["naive_pred"])
    naive_rmse_3yr = train.rmse(far["y_true"], far["naive_pred"])
    gof_near = 1.0 - (rmse_1yr / naive_rmse_1yr if naive_rmse_1yr else np.inf)
    gof_far = 1.0 - (rmse_3yr / naive_rmse_3yr if naive_rmse_3yr else np.inf)
    theil_u = rmse_1yr / naive_rmse_1yr if naive_rmse_1yr else np.nan
    overall_mae = train.mae(near["y_true"], near[prediction_column])

    return {
        "gof_validation_near": float(gof_near),
        "gof_validation_far": float(gof_far),
        "rmse_1yr": float(rmse_1yr),
        "rmse_3yr": float(rmse_3yr),
        "naive_rmse_1yr": float(naive_rmse_1yr),
        "naive_rmse_3yr": float(naive_rmse_3yr),
        "rmse": float(rmse_1yr),
        "mae": float(overall_mae),
        "theil_u": float(theil_u),
    }


def load_component_forecasts(components: list[dict[str, Any]]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component in components:
        forecast = pd.read_csv(component["run_dir"] / "forecast_120q.csv")
        forecast["date"] = pd.to_datetime(forecast["date"])
        key = component["component_name"]
        renamed = forecast.rename(
            columns={
                "hpi_forecast": f"{key}__forecast",
                "hpi_lower_90": f"{key}__lower_90",
                "hpi_upper_90": f"{key}__upper_90",
                "hpi_lower_50": f"{key}__lower_50",
                "hpi_upper_50": f"{key}__upper_50",
            }
        )
        cols = ["date", f"{key}__forecast", f"{key}__lower_90", f"{key}__upper_90", f"{key}__lower_50", f"{key}__upper_50"]
        if merged is None:
            merged = renamed[cols].copy()
        else:
            merged = merged.merge(renamed[cols], on="date", how="inner")
    if merged is None or merged.empty:
        raise RuntimeError("No forecast_120q.csv files were available for ensemble construction.")
    return merged.sort_values("date").reset_index(drop=True)


def forecast_plausibility(series: pd.Series) -> bool:
    changes = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    start_value = float(series.iloc[0]) if len(series) else np.nan
    end_value = float(series.iloc[-1]) if len(series) else np.nan
    if len(series) > 1 and np.isfinite(start_value) and start_value != 0.0:
        cumulative = (end_value / start_value) - 1.0
    else:
        cumulative = np.nan
    return bool(
        (series > 0).all()
        and (changes.abs() < 0.15).all()
        and np.isfinite(cumulative)
        and (-0.20 <= cumulative <= 5.0)
        and (float(series.std()) > 0.0)
    )


def component_metric_average(components: list[dict[str, Any]], field: str) -> float:
    values = [float(component["metrics"][field]) for component in components]
    return float(np.mean(values))


def component_param_total(components: list[dict[str, Any]]) -> int:
    return int(sum(int(component["metrics"]["n_params"]) for component in components))


def build_simple_average(validation: pd.DataFrame, pred_columns: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    return validation[pred_columns].mean(axis=1), {"weights": {column: 1.0 / len(pred_columns) for column in pred_columns}}


def build_inverse_rmse(validation: pd.DataFrame, components: list[dict[str, Any]], pred_columns: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    inverse = np.asarray(
        [1.0 / float(component["metrics"]["rmse_1yr"]) for component in components],
        dtype=float,
    )
    weights = inverse / inverse.sum()
    prediction = validation[pred_columns].to_numpy() @ weights
    return pd.Series(prediction, index=validation.index), {
        "weights": {column: float(weight) for column, weight in zip(pred_columns, weights)}
    }


def build_ridge_stacking(validation: pd.DataFrame, pred_columns: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    frame = validation.copy()
    origins = pd.Series(pd.to_datetime(frame["origin_date"]).astype(str))
    unique_origins = origins.nunique()
    n_splits = min(5, unique_origins)
    if n_splits < 2:
        raise RuntimeError("Need at least two backtest origins for ridge stacking.")

    x_values = frame[pred_columns].to_numpy(dtype=float)
    y_values = frame["y_true"].to_numpy(dtype=float)
    oof = np.full(len(frame), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    alphas = np.logspace(-3, 3, 13)
    for train_idx, test_idx in splitter.split(x_values, y_values, groups=origins):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas)),
            ]
        )
        model.fit(x_values[train_idx], y_values[train_idx])
        oof[test_idx] = model.predict(x_values[test_idx])

    if np.isnan(oof).any():
        raise RuntimeError("Ridge stacking failed to produce out-of-fold predictions.")

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ]
    )
    final_model.fit(x_values, y_values)
    scaler = final_model.named_steps["scaler"]
    ridge = final_model.named_steps["ridge"]
    transformed_coef = ridge.coef_ / scaler.scale_
    abs_weights = np.abs(transformed_coef)
    abs_weights = abs_weights / abs_weights.sum() if abs_weights.sum() else np.repeat(1.0 / len(pred_columns), len(pred_columns))

    return pd.Series(oof, index=validation.index), {
        "final_model": final_model,
        "coefficients": {column: float(value) for column, value in zip(pred_columns, transformed_coef)},
        "abs_interval_weights": {column: float(value) for column, value in zip(pred_columns, abs_weights)},
        "alpha": float(ridge.alpha_),
        "intercept": float(ridge.intercept_ - np.sum((scaler.mean_ / scaler.scale_) * ridge.coef_)),
    }


def combine_forecasts(
    future_frame: pd.DataFrame,
    pred_columns: list[str],
    method: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    base = pd.DataFrame({"date": future_frame["date"], "hpi_actual": np.nan})
    if method == "simple_average":
        weights = metadata["weights"]
        forecast = sum(future_frame[column.replace("y_pred", "forecast")] * weights[column] for column in pred_columns)
        lower_90 = sum(future_frame[column.replace("y_pred", "lower_90")] * weights[column] for column in pred_columns)
        upper_90 = sum(future_frame[column.replace("y_pred", "upper_90")] * weights[column] for column in pred_columns)
        lower_50 = sum(future_frame[column.replace("y_pred", "lower_50")] * weights[column] for column in pred_columns)
        upper_50 = sum(future_frame[column.replace("y_pred", "upper_50")] * weights[column] for column in pred_columns)
    elif method == "inverse_rmse":
        weights = metadata["weights"]
        forecast = sum(future_frame[column.replace("y_pred", "forecast")] * weights[column] for column in pred_columns)
        lower_90 = sum(future_frame[column.replace("y_pred", "lower_90")] * weights[column] for column in pred_columns)
        upper_90 = sum(future_frame[column.replace("y_pred", "upper_90")] * weights[column] for column in pred_columns)
        lower_50 = sum(future_frame[column.replace("y_pred", "lower_50")] * weights[column] for column in pred_columns)
        upper_50 = sum(future_frame[column.replace("y_pred", "upper_50")] * weights[column] for column in pred_columns)
    elif method == "ridge_stacking":
        model = metadata["final_model"]
        forecast_cols = [column.replace("y_pred", "forecast") for column in pred_columns]
        point_forecast = model.predict(future_frame[forecast_cols].to_numpy(dtype=float))
        interval_weights = metadata["abs_interval_weights"]
        width_90 = sum(
            ((future_frame[column.replace("y_pred", "upper_90")] - future_frame[column.replace("y_pred", "lower_90")]) / 2.0)
            * interval_weights[column]
            for column in pred_columns
        )
        width_50 = sum(
            ((future_frame[column.replace("y_pred", "upper_50")] - future_frame[column.replace("y_pred", "lower_50")]) / 2.0)
            * interval_weights[column]
            for column in pred_columns
        )
        forecast = pd.Series(point_forecast, index=future_frame.index)
        lower_90 = forecast - width_90
        upper_90 = forecast + width_90
        lower_50 = forecast - width_50
        upper_50 = forecast + width_50
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported ensemble method: {method}")

    base["hpi_forecast"] = np.asarray(forecast, dtype=float)
    base["hpi_lower_90"] = np.asarray(lower_90, dtype=float)
    base["hpi_upper_90"] = np.asarray(upper_90, dtype=float)
    base["hpi_lower_50"] = np.asarray(lower_50, dtype=float)
    base["hpi_upper_50"] = np.asarray(upper_50, dtype=float)
    return base


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def persist_trial(
    experiment_id: str,
    trial_id: str,
    method: str,
    components: list[dict[str, Any]],
    validation: pd.DataFrame,
    future_forecast: pd.DataFrame,
    ensemble_metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    experiment_dir = ensure_experiment_dirs(experiment_id)
    output_dir = experiment_dir / "runs" / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(__file__), output_dir / "ensemble_snapshot.py")
    future_forecast.to_csv(output_dir / "forecast_120q.csv", index=False)
    validation.to_parquet(output_dir / "validation_predictions.parquet", index=False)

    spec = {
        "experiment_id": experiment_id,
        "trial_id": trial_id,
        "model_class": "Model Ensemble",
        "champion_eligible": True,
        "dependency_scope": "ensemble",
        "method": method,
        "components": [
            {
                "component_name": component["component_name"],
                "experiment_id": component["experiment_id"],
                "trial_id": component["trial_id"],
                "description": component["metrics"]["description"],
            }
            for component in components
        ],
        "metadata": ensemble_metadata,
    }
    write_json(output_dir / "spec.json", spec)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "stdout.log").write_text(f"Completed ensemble method {method}.\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")

    results_row = {
        "experiment_id": experiment_id,
        "trial_id": trial_id,
        "accepted_commit": "",
        "model_class": "Model Ensemble",
        "champion_eligible": True,
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
    train.append_tsv(experiment_dir / "results.tsv", train.RESULTS_HEADER, results_row)
    train.append_tsv(train.ROOT_RESULTS_PATH, train.RESULTS_HEADER, results_row)


def update_experiment_leaderboard(experiment_id: str) -> None:
    experiment_dir = ROOT / "experiments" / experiment_id
    results_path = experiment_dir / "results.tsv"
    results = pd.read_csv(results_path, sep="\t")
    ok_rows = results.loc[results["status"] == "ok"].copy()
    if ok_rows.empty:
        return
    best = ok_rows.sort_values("gof_composite", ascending=False).iloc[0]
    leaderboard_row = {
        "rank": 1,
        "experiment_id": experiment_id,
        "model_class": "Model Ensemble",
        "champion_eligible": True,
        "best_trial_id": best["trial_id"],
        "best_commit": "",
        "gof_composite": float(best["gof_composite"]),
        "gof_insample": float(best["gof_insample"]),
        "gof_val_near": float(best["gof_val_near"]),
        "gof_val_far": float(best["gof_val_far"]),
        "rmse_1yr": float(best["rmse"]),
        "rmse_3yr": float(
            read_json(Path(best["artifact_dir"]) / "metrics.json")["rmse_3yr"]
        ),
        "n_params": int(best["n_params"]),
        "n_trials": int(len(results)),
        "description": str(best["description"]),
    }
    train.upsert_leaderboard(experiment_dir / "leaderboard.tsv", leaderboard_row)
    train.upsert_leaderboard(train.ROOT_LEADERBOARD_PATH, leaderboard_row)


def prepare_future_matrix(future_forecast: pd.DataFrame, pred_columns: list[str]) -> pd.DataFrame:
    prepared = future_forecast.copy()
    for column in pred_columns:
        component = column.replace("__y_pred", "")
        prepared[column.replace("y_pred", "forecast")] = prepared[f"{component}__forecast"]
        prepared[column.replace("y_pred", "lower_90")] = prepared[f"{component}__lower_90"]
        prepared[column.replace("y_pred", "upper_90")] = prepared[f"{component}__upper_90"]
        prepared[column.replace("y_pred", "lower_50")] = prepared[f"{component}__lower_50"]
        prepared[column.replace("y_pred", "upper_50")] = prepared[f"{component}__upper_50"]
    return prepared


def run_method(
    experiment_id: str,
    trial_index: int,
    method: str,
    components: list[dict[str, Any]],
    validation: pd.DataFrame,
    future_forecast: pd.DataFrame,
    pred_columns: list[str],
) -> None:
    if method == "simple_average":
        ensemble_pred, method_metadata = build_simple_average(validation, pred_columns)
        ensemble_param_count = 0
    elif method == "inverse_rmse":
        ensemble_pred, method_metadata = build_inverse_rmse(validation, components, pred_columns)
        ensemble_param_count = len(pred_columns)
    elif method == "ridge_stacking":
        ensemble_pred, method_metadata = build_ridge_stacking(validation, pred_columns)
        ensemble_param_count = len(pred_columns) + 1
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported method {method}.")

    validation_frame = validation.copy()
    validation_frame["y_pred"] = np.asarray(ensemble_pred, dtype=float)
    metrics = validation_metrics(validation_frame, "y_pred")
    future_matrix = prepare_future_matrix(future_forecast, pred_columns)
    ensemble_future = combine_forecasts(future_matrix, pred_columns, method, method_metadata)

    component_insample = component_metric_average(components, "gof_insample")
    component_diag = component_metric_average(components, "gof_diagnostic")
    plausibility = 1.0 if forecast_plausibility(ensemble_future["hpi_forecast"]) else 0.0
    gof_diag = float((component_diag + plausibility) / 2.0)
    gof_composite = 0.40 * component_insample + 0.35 * metrics["gof_validation_near"] + 0.15 * metrics["gof_validation_far"] + 0.10 * gof_diag

    component_names = [component["component_name"] for component in components]
    metrics_payload = {
        "experiment_id": experiment_id,
        "trial_id": f"{experiment_id}_{trial_index:03d}",
        "model_class": "Model Ensemble",
        "champion_eligible": True,
        "description": f"{method} ensemble of {', '.join(component_names)}",
        "gof_composite": float(gof_composite),
        "gof_insample": float(component_insample),
        "gof_validation_near": float(metrics["gof_validation_near"]),
        "gof_validation_far": float(metrics["gof_validation_far"]),
        "gof_diagnostic": float(gof_diag),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "theil_u": float(metrics["theil_u"]),
        "directional_accuracy": float(
            (
                np.sign(validation_frame["y_true"].diff().dropna())
                == np.sign(validation_frame["y_pred"].diff().dropna())
            ).mean()
        ),
        "n_params": int(component_param_total(components) + ensemble_param_count),
        "status": "ok",
        "diagnostics_passed": ["component_diagnostics_mean", "forecast_plausibility"] if plausibility else ["component_diagnostics_mean"],
        "diagnostics_failed": [] if plausibility else ["forecast_plausibility"],
        "error_summary": "",
        "rmse_1yr": float(metrics["rmse_1yr"]),
        "rmse_3yr": float(metrics["rmse_3yr"]),
        "insample_rmse": np.nan,
        "naive_insample_rmse": np.nan,
        "component_models": [
            {
                "component_name": component["component_name"],
                "experiment_id": component["experiment_id"],
                "trial_id": component["trial_id"],
            }
            for component in components
        ],
    }

    persist_trial(
        experiment_id=experiment_id,
        trial_id=metrics_payload["trial_id"],
        method=method,
        components=components,
        validation=validation_frame[["origin_date", "forecast_date", "horizon_q", "y_true", "y_pred"]],
        future_forecast=ensemble_future,
        ensemble_metadata=method_metadata,
        metrics=metrics_payload,
    )


def main() -> int:
    args = parse_args()
    experiment_dir = ensure_experiment_dirs(args.experiment_id)
    components = [component_artifacts(row) for row in select_component_rows(args.component_classes)]
    validation = attach_naive_predictions(merge_validation_frames(components))
    pred_columns = [f"{component['component_name']}__y_pred" for component in components]
    validation = validation.rename(columns={component["component_name"]: f"{component['component_name']}__y_pred" for component in components})
    future_forecast = load_component_forecasts(components)

    for index, method in enumerate(METHODS, start=1):
        run_method(
            experiment_id=args.experiment_id,
            trial_index=index,
            method=method,
            components=components,
            validation=validation,
            future_forecast=future_forecast,
            pred_columns=pred_columns,
        )

    update_experiment_leaderboard(args.experiment_id)
    (experiment_dir / "notes.md").write_text(
        "Components:\n" + "\n".join(
            f"- {component['component_name']}: {component['experiment_id']} / {component['trial_id']}"
            for component in components
        )
        + "\nMethods:\n- simple_average\n- inverse_rmse\n- ridge_stacking\n",
        encoding="utf-8",
    )
    print(f"Completed ensemble experiment {args.experiment_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
