#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-spec-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def component_importance(component: dict[str, Any], search_frame: pd.DataFrame) -> dict[str, Any]:
    run_dir = ROOT / "experiments" / component["experiment_id"] / "runs" / component["trial_id"]
    run_spec = read_json(run_dir / "spec.json")
    train.set_active_spec(run_spec["spec"])
    model_class = run_spec["model_class"]
    if model_class == "Gradient Boosting":
        fitted = train.fit_gradient_boosting(search_frame.copy())
    elif model_class == "XGBoost":
        fitted = train.fit_xgboost(search_frame.copy())
    else:
        raise RuntimeError(f"Feature importance generator does not support {model_class}.")

    importances = []
    for feature_name in fitted.feature_names:
        key = f"importance_{feature_name}"
        value = float(fitted.params.get(key, np.nan))
        if np.isfinite(value):
            importances.append({"feature": feature_name, "importance": value})
    importances.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "model_class": model_class,
        "experiment_id": component["experiment_id"],
        "trial_id": component["trial_id"],
        "description": component.get("description", ""),
        "target_lags": list(fitted.target_lags),
        "exogenous": list(fitted.exogenous),
        "feature_importance": importances,
    }


def main() -> int:
    args = parse_args()
    finalized_spec = read_json(Path(args.finalized_spec_path))
    search_frame, _, _, _ = train.load_panels()
    search_frame = train.filter_sample(search_frame)

    components = [component_importance(component, search_frame) for component in finalized_spec["components"]]
    ensemble_weights = finalized_spec.get("metadata", {}).get("weights", {})
    payload = {
        "model_class": finalized_spec["model_class"],
        "method": finalized_spec.get("method"),
        "component_weights": ensemble_weights,
        "components": components,
    }
    write_json(Path(args.output_path), payload)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
