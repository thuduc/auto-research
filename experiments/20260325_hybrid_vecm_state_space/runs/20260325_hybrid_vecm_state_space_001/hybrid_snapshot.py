#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train


ROOT = Path(__file__).resolve().parent
METHODS = ["hard_switch_12", "smooth_blend_9_16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def ensure_experiment_dir(experiment_id: str) -> Path:
    experiment_dir = ROOT / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def leaderboard_best(model_class: str) -> pd.Series:
    board = pd.read_csv(ROOT / "leaderboard.tsv", sep="\t")
    board["gof_composite"] = pd.to_numeric(board["gof_composite"], errors="coerce")
    subset = board.loc[board["model_class"] == model_class].sort_values("gof_composite", ascending=False)
    if subset.empty:
        raise RuntimeError(f"No leaderboard entry found for model class {model_class}.")
    return subset.iloc[0]


def component_info(row: pd.Series) -> dict[str, Any]:
    experiment_id = str(row["experiment_id"])
    trial_id = str(row["best_trial_id"])
    run_dir = ROOT / "experiments" / experiment_id / "runs" / trial_id
    return {
        "model_class": str(row["model_class"]),
        "experiment_id": experiment_id,
        "trial_id": trial_id,
        "run_dir": run_dir,
        "metrics": read_json(run_dir / "metrics.json"),
    }


def load_validation(component: dict[str, Any], label: str) -> pd.DataFrame:
    frame = pd.read_parquet(component["run_dir"] / "validation_predictions.parquet")
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    return frame[["origin_date", "forecast_date", "horizon_q", "y_true", "y_pred"]].rename(
        columns={"y_pred": f"{label}_pred"}
    )


def load_forecast(component: dict[str, Any], label: str) -> pd.DataFrame:
    frame = pd.read_csv(component["run_dir"] / "forecast_120q.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.rename(
        columns={
            "hpi_forecast": f"{label}_forecast",
            "hpi_lower_90": f"{label}_lower_90",
            "hpi_upper_90": f"{label}_upper_90",
            "hpi_lower_50": f"{label}_lower_50",
            "hpi_upper_50": f"{label}_upper_50",
        }
    )


def weight_schedule(method: str, horizons: pd.Series) -> pd.DataFrame:
    horizon_values = horizons.to_numpy(dtype=int)
    if method == "hard_switch_12":
        state_weight = np.where(horizon_values <= 12, 1.0, 0.0)
    elif method == "smooth_blend_9_16":
        state_weight = np.where(
            horizon_values <= 8,
            1.0,
            np.where(
                horizon_values >= 17,
                0.0,
                (17.0 - horizon_values) / 9.0,
            ),
        )
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported hybrid method: {method}")
    vecm_weight = 1.0 - state_weight
    return pd.DataFrame({"state_weight": state_weight, "vecm_weight": vecm_weight}, index=horizons.index)


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


def score_validation(frame: pd.DataFrame) -> dict[str, float]:
    near = frame.loc[frame["horizon_q"] <= 12].copy()
    far = frame.loc[frame["horizon_q"] > 12].copy()
    naive_lookup = build_naive_lookup()
    frame = frame.copy()
    frame["naive_pred"] = [
        naive_lookup[(pd.Timestamp(origin), pd.Timestamp(forecast))]
        for origin, forecast in zip(frame["origin_date"], frame["forecast_date"])
    ]
    near["naive_pred"] = frame.loc[near.index, "naive_pred"]
    far["naive_pred"] = frame.loc[far.index, "naive_pred"]

    rmse_1yr = train.rmse(near["y_true"], near["y_pred"])
    rmse_3yr = train.rmse(far["y_true"], far["y_pred"])
    naive_rmse_1yr = train.rmse(near["y_true"], near["naive_pred"])
    naive_rmse_3yr = train.rmse(far["y_true"], far["naive_pred"])
    gof_near = 1.0 - (rmse_1yr / naive_rmse_1yr if naive_rmse_1yr else np.inf)
    gof_far = 1.0 - (rmse_3yr / naive_rmse_3yr if naive_rmse_3yr else np.inf)
    theil_u = rmse_1yr / naive_rmse_1yr if naive_rmse_1yr else np.nan
    mae = train.mae(near["y_true"], near["y_pred"])
    directional = float(
        (
            np.sign(frame["y_true"].diff().dropna())
            == np.sign(frame["y_pred"].diff().dropna())
        ).mean()
    )
    return {
        "gof_validation_near": float(gof_near),
        "gof_validation_far": float(gof_far),
        "rmse_1yr": float(rmse_1yr),
        "rmse_3yr": float(rmse_3yr),
        "rmse": float(rmse_1yr),
        "mae": float(mae),
        "theil_u": float(theil_u),
        "directional_accuracy": directional,
    }


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


def build_hybrid_validation(vecm_component: dict[str, Any], state_component: dict[str, Any], method: str) -> pd.DataFrame:
    vecm = load_validation(vecm_component, "vecm")
    state = load_validation(state_component, "state")
    merged = vecm.merge(state, on=["origin_date", "forecast_date", "horizon_q", "y_true"], how="inner")
    weights = weight_schedule(method, merged["horizon_q"])
    merged["vecm_weight"] = weights["vecm_weight"]
    merged["state_weight"] = weights["state_weight"]
    merged["y_pred"] = merged["vecm_pred"] * merged["vecm_weight"] + merged["state_pred"] * merged["state_weight"]
    return merged.sort_values(["origin_date", "forecast_date"]).reset_index(drop=True)


def build_hybrid_forecast(vecm_component: dict[str, Any], state_component: dict[str, Any], method: str) -> pd.DataFrame:
    vecm = load_forecast(vecm_component, "vecm")
    state = load_forecast(state_component, "state")
    merged = vecm.merge(state, on=["date", "hpi_actual"], how="inner")
    horizons = pd.Series(np.arange(1, len(merged) + 1), index=merged.index)
    weights = weight_schedule(method, horizons)
    merged["vecm_weight"] = weights["vecm_weight"]
    merged["state_weight"] = weights["state_weight"]
    for suffix in ["forecast", "lower_90", "upper_90", "lower_50", "upper_50"]:
        merged[f"hpi_{suffix}"] = (
            merged[f"vecm_{suffix}"] * merged["vecm_weight"] + merged[f"state_{suffix}"] * merged["state_weight"]
        )
    return merged[["date", "hpi_actual", "hpi_forecast", "hpi_lower_90", "hpi_upper_90", "hpi_lower_50", "hpi_upper_50"]]


def weighted_component_average(method: str, vecm_value: float, state_value: float) -> float:
    horizons = pd.Series(np.arange(1, 41))
    weights = weight_schedule(method, horizons)
    vecm_weight = float(weights["vecm_weight"].mean())
    state_weight = float(weights["state_weight"].mean())
    return vecm_weight * vecm_value + state_weight * state_value


def persist_trial(
    experiment_id: str,
    trial_id: str,
    method: str,
    vecm_component: dict[str, Any],
    state_component: dict[str, Any],
    validation: pd.DataFrame,
    forecast: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    experiment_dir = ensure_experiment_dir(experiment_id)
    output_dir = experiment_dir / "runs" / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(__file__), output_dir / "hybrid_snapshot.py")
    validation[["origin_date", "forecast_date", "horizon_q", "y_true", "y_pred"]].to_parquet(
        output_dir / "validation_predictions.parquet",
        index=False,
    )
    forecast.to_csv(output_dir / "forecast_120q.csv", index=False)

    spec = {
        "experiment_id": experiment_id,
        "trial_id": trial_id,
        "model_class": "Horizon Hybrid",
        "champion_eligible": True,
        "dependency_scope": "ensemble",
        "method": method,
        "components": [
            {
                "role": "long_horizon",
                "model_class": vecm_component["model_class"],
                "experiment_id": vecm_component["experiment_id"],
                "trial_id": vecm_component["trial_id"],
                "description": vecm_component["metrics"]["description"],
            },
            {
                "role": "near_horizon",
                "model_class": state_component["model_class"],
                "experiment_id": state_component["experiment_id"],
                "trial_id": state_component["trial_id"],
                "description": state_component["metrics"]["description"],
            },
        ],
    }
    write_json(output_dir / "spec.json", spec)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "stdout.log").write_text(f"Completed hybrid method {method}.\n", encoding="utf-8")
    (output_dir / "stderr.log").write_text("", encoding="utf-8")

    results_row = {
        "experiment_id": experiment_id,
        "trial_id": trial_id,
        "accepted_commit": "",
        "model_class": "Horizon Hybrid",
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
    results = pd.read_csv(experiment_dir / "results.tsv", sep="\t")
    ok_rows = results.loc[results["status"] == "ok"].copy()
    if ok_rows.empty:
        return
    best = ok_rows.sort_values("gof_composite", ascending=False).iloc[0]
    metrics = read_json(Path(best["artifact_dir"]) / "metrics.json")
    leaderboard_row = {
        "rank": 1,
        "experiment_id": experiment_id,
        "model_class": "Horizon Hybrid",
        "champion_eligible": True,
        "best_trial_id": best["trial_id"],
        "best_commit": "",
        "gof_composite": float(best["gof_composite"]),
        "gof_insample": float(best["gof_insample"]),
        "gof_val_near": float(best["gof_val_near"]),
        "gof_val_far": float(best["gof_val_far"]),
        "rmse_1yr": float(best["rmse"]),
        "rmse_3yr": float(metrics["rmse_3yr"]),
        "n_params": int(best["n_params"]),
        "n_trials": int(len(results)),
        "description": str(best["description"]),
    }
    train.upsert_leaderboard(experiment_dir / "leaderboard.tsv", leaderboard_row)
    train.upsert_leaderboard(train.ROOT_LEADERBOARD_PATH, leaderboard_row)


def run_method(
    experiment_id: str,
    trial_index: int,
    method: str,
    vecm_component: dict[str, Any],
    state_component: dict[str, Any],
) -> None:
    validation = build_hybrid_validation(vecm_component, state_component, method)
    forecast = build_hybrid_forecast(vecm_component, state_component, method)
    scored = score_validation(validation)

    insample = weighted_component_average(
        method,
        float(vecm_component["metrics"]["gof_insample"]),
        float(state_component["metrics"]["gof_insample"]),
    )
    component_diag = weighted_component_average(
        method,
        float(vecm_component["metrics"]["gof_diagnostic"]),
        float(state_component["metrics"]["gof_diagnostic"]),
    )
    plausibility = 1.0 if forecast_plausibility(forecast["hpi_forecast"]) else 0.0
    gof_diag = float((component_diag + plausibility) / 2.0)
    gof_composite = 0.40 * insample + 0.35 * scored["gof_validation_near"] + 0.15 * scored["gof_validation_far"] + 0.10 * gof_diag

    metrics = {
        "experiment_id": experiment_id,
        "trial_id": f"{experiment_id}_{trial_index:03d}",
        "model_class": "Horizon Hybrid",
        "champion_eligible": True,
        "description": f"{method} hybrid of State-Space (near) and VECM (far)",
        "gof_composite": float(gof_composite),
        "gof_insample": float(insample),
        "gof_validation_near": float(scored["gof_validation_near"]),
        "gof_validation_far": float(scored["gof_validation_far"]),
        "gof_diagnostic": float(gof_diag),
        "rmse": float(scored["rmse"]),
        "mae": float(scored["mae"]),
        "theil_u": float(scored["theil_u"]),
        "directional_accuracy": float(scored["directional_accuracy"]),
        "n_params": int(vecm_component["metrics"]["n_params"] + state_component["metrics"]["n_params"]),
        "status": "ok",
        "diagnostics_passed": ["component_diagnostics_weighted", "forecast_plausibility"] if plausibility else ["component_diagnostics_weighted"],
        "diagnostics_failed": [] if plausibility else ["forecast_plausibility"],
        "error_summary": "",
        "rmse_1yr": float(scored["rmse_1yr"]),
        "rmse_3yr": float(scored["rmse_3yr"]),
        "insample_rmse": np.nan,
        "naive_insample_rmse": np.nan,
        "components": [
            {"role": "long_horizon", "experiment_id": vecm_component["experiment_id"], "trial_id": vecm_component["trial_id"]},
            {"role": "near_horizon", "experiment_id": state_component["experiment_id"], "trial_id": state_component["trial_id"]},
        ],
    }

    persist_trial(
        experiment_id=experiment_id,
        trial_id=metrics["trial_id"],
        method=method,
        vecm_component=vecm_component,
        state_component=state_component,
        validation=validation,
        forecast=forecast,
        metrics=metrics,
    )


def main() -> int:
    args = parse_args()
    experiment_dir = ensure_experiment_dir(args.experiment_id)
    vecm_component = component_info(leaderboard_best("VECM"))
    state_component = component_info(leaderboard_best("State-Space"))

    for index, method in enumerate(METHODS, start=1):
        run_method(
            experiment_id=args.experiment_id,
            trial_index=index,
            method=method,
            vecm_component=vecm_component,
            state_component=state_component,
        )

    update_experiment_leaderboard(args.experiment_id)
    (experiment_dir / "notes.md").write_text(
        "Components:\n"
        f"- VECM: {vecm_component['experiment_id']} / {vecm_component['trial_id']}\n"
        f"- State-Space: {state_component['experiment_id']} / {state_component['trial_id']}\n"
        "Methods:\n"
        "- hard_switch_12\n"
        "- smooth_blend_9_16\n",
        encoding="utf-8",
    )
    print(f"Completed hybrid experiment {args.experiment_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
