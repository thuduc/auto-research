#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import train


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment-id", required=True)
    parser.add_argument("--source-trial-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--trial-id", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> int:
    args = parse_args()
    source_dir = ROOT / "experiments" / args.source_experiment_id / "runs" / args.source_trial_id
    if not source_dir.exists():
        raise RuntimeError(f"Missing source run directory: {source_dir}")

    target_experiment_dir = ROOT / "experiments" / args.experiment_id
    target_run_dir = target_experiment_dir / "runs" / args.trial_id
    target_run_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "forecast_120q.csv",
        "validation_predictions.parquet",
        "ensemble_snapshot.py",
        "stdout.log",
        "stderr.log",
    ]:
        shutil.copy2(source_dir / name, target_run_dir / name)

    source_spec = read_json(source_dir / "spec.json")
    source_metrics = read_json(source_dir / "metrics.json")

    finalized_spec = dict(source_spec)
    finalized_spec["experiment_id"] = args.experiment_id
    finalized_spec["trial_id"] = args.trial_id
    finalized_spec["finalized_from"] = {
        "experiment_id": args.source_experiment_id,
        "trial_id": args.source_trial_id,
    }
    write_json(target_run_dir / "spec.json", finalized_spec)

    finalized_metrics = dict(source_metrics)
    finalized_metrics["experiment_id"] = args.experiment_id
    finalized_metrics["trial_id"] = args.trial_id
    finalized_metrics["description"] = f"Finalized {source_metrics['description']}"
    write_json(target_run_dir / "metrics.json", finalized_metrics)

    results_row = {
        "experiment_id": args.experiment_id,
        "trial_id": args.trial_id,
        "accepted_commit": "",
        "model_class": "Model Ensemble",
        "champion_eligible": True,
        "gof_composite": finalized_metrics["gof_composite"],
        "gof_insample": finalized_metrics["gof_insample"],
        "gof_val_near": finalized_metrics["gof_validation_near"],
        "gof_val_far": finalized_metrics["gof_validation_far"],
        "gof_diag": finalized_metrics["gof_diagnostic"],
        "rmse": finalized_metrics["rmse"],
        "mae": finalized_metrics["mae"],
        "theil_u": finalized_metrics["theil_u"],
        "n_params": finalized_metrics["n_params"],
        "status": finalized_metrics["status"],
        "description": finalized_metrics["description"],
        "artifact_dir": str(target_run_dir),
        "error_summary": finalized_metrics["error_summary"],
    }
    train.append_tsv(target_experiment_dir / "results.tsv", train.RESULTS_HEADER, results_row)
    train.append_tsv(train.ROOT_RESULTS_PATH, train.RESULTS_HEADER, results_row)

    leaderboard_row = {
        "rank": 1,
        "experiment_id": args.experiment_id,
        "model_class": "Model Ensemble",
        "champion_eligible": True,
        "best_trial_id": args.trial_id,
        "best_commit": "",
        "gof_composite": float(finalized_metrics["gof_composite"]),
        "gof_insample": float(finalized_metrics["gof_insample"]),
        "gof_val_near": float(finalized_metrics["gof_validation_near"]),
        "gof_val_far": float(finalized_metrics["gof_validation_far"]),
        "rmse_1yr": float(finalized_metrics["rmse_1yr"]),
        "rmse_3yr": float(finalized_metrics["rmse_3yr"]),
        "n_params": int(finalized_metrics["n_params"]),
        "n_trials": 1,
        "description": str(finalized_metrics["description"]),
    }
    train.upsert_leaderboard(target_experiment_dir / "leaderboard.tsv", leaderboard_row)
    train.upsert_leaderboard(train.ROOT_LEADERBOARD_PATH, leaderboard_row)

    notes = (
        "Finalized ensemble champion\n"
        f"- source_experiment_id: {args.source_experiment_id}\n"
        f"- source_trial_id: {args.source_trial_id}\n"
        f"- method: {source_spec['method']}\n"
        f"- components: {json.dumps(source_spec['components'])}\n"
    )
    (target_experiment_dir / "notes.md").write_text(notes, encoding="utf-8")
    print(f"Finalized ensemble champion into {args.experiment_id}/{args.trial_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
