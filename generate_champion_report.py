#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-run-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    errors = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float))))


def main() -> int:
    args = parse_args()
    run_dir = Path(args.finalized_run_dir)
    spec = read_json(run_dir / "spec.json")
    metrics = read_json(run_dir / "metrics.json")
    manifest = read_json(ROOT / "data" / "processed" / "dataset_manifest.json")
    holdout = pd.read_parquet(ROOT / "data" / "processed" / "holdout_panel.parquet")
    holdout["date"] = pd.to_datetime(holdout["date"])
    forecast = pd.read_csv(run_dir / "forecast_120q.csv")
    forecast["date"] = pd.to_datetime(forecast["date"])

    holdout_join = holdout[["date", "hpi"]].merge(
        forecast[["date", "hpi_forecast", "hpi_lower_90", "hpi_upper_90", "hpi_lower_50", "hpi_upper_50"]],
        on="date",
        how="left",
    )
    holdout_join = holdout_join.rename(columns={"hpi": "hpi_actual"})
    holdout_join["inside_90"] = (
        (holdout_join["hpi_actual"] >= holdout_join["hpi_lower_90"])
        & (holdout_join["hpi_actual"] <= holdout_join["hpi_upper_90"])
    )
    holdout_join["inside_50"] = (
        (holdout_join["hpi_actual"] >= holdout_join["hpi_lower_50"])
        & (holdout_join["hpi_actual"] <= holdout_join["hpi_upper_50"])
    )

    holdout_metrics = {
        "rows": int(len(holdout_join)),
        "rmse": rmse(holdout_join["hpi_actual"], holdout_join["hpi_forecast"]),
        "mae": mae(holdout_join["hpi_actual"], holdout_join["hpi_forecast"]),
        "coverage_90": float(holdout_join["inside_90"].mean()),
        "coverage_50": float(holdout_join["inside_50"].mean()),
    }

    scenario_summary = {
        "baseline": {
            "implemented": False,
            "reason": "Current finalized ensemble artifact contains baseline-only forecast output; adverse and severely adverse scenario generation has not yet been implemented for ensemble champions.",
        },
        "adverse": {"implemented": False},
        "severely_adverse": {"implemented": False},
    }

    leaderboard = pd.read_csv(ROOT / "leaderboard.tsv", sep="\t")
    top_rows = leaderboard[["experiment_id", "model_class", "gof_composite", "description"]].head(10).to_dict(orient="records")

    payload = {
        "champion": {
            "experiment_id": spec["experiment_id"],
            "trial_id": spec["trial_id"],
            "model_class": spec["model_class"],
            "description": metrics["description"],
            "method": spec.get("method"),
            "components": spec.get("components", []),
            "weights": spec.get("metadata", {}).get("weights", {}),
        },
        "sample_summary": {
            "search_start": manifest.get("search_start"),
            "search_end": manifest.get("search_end"),
            "holdout_start": manifest.get("holdout_start"),
            "holdout_end": manifest.get("holdout_end"),
            "backtest_origins": manifest.get("backtest_origins", []),
        },
        "validation_metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_predictions": holdout_join.to_dict(orient="records"),
        "scenario_summary": scenario_summary,
        "benchmark_summary": top_rows,
        "diagnostics": {
            "diagnostics_passed": metrics.get("diagnostics_passed", []),
            "diagnostics_failed": metrics.get("diagnostics_failed", []),
        },
    }
    write_json(Path(args.output_path), payload)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
