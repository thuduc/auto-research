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
    parser.add_argument("--champion-report-path", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_component_fit(component: dict[str, Any], search_frame: pd.DataFrame) -> tuple[dict[str, Any], Any]:
    run_dir = ROOT / "experiments" / component["experiment_id"] / "runs" / component["trial_id"]
    run_spec = read_json(run_dir / "spec.json")
    train.set_active_spec(run_spec["spec"])
    model_class = run_spec["model_class"]
    if model_class == "Gradient Boosting":
        fitted = train.fit_gradient_boosting(search_frame.copy())
        spec = run_spec["spec"]["gradient_boosting"]
    elif model_class == "XGBoost":
        fitted = train.fit_xgboost(search_frame.copy())
        spec = run_spec["spec"]["xgboost"]
    else:
        raise RuntimeError(f"Unsupported ensemble scenario component: {model_class}")
    return spec, fitted


def stressed_future_exog(
    search_frame: pd.DataFrame,
    variable_config: dict[str, Any],
    horizon: int,
    scenario: str,
) -> pd.DataFrame:
    baseline = train.build_future_exog(search_frame, variable_config, horizon=horizon)
    if scenario == "baseline":
        return baseline

    stressed = baseline.copy()
    trailing_20q = search_frame.tail(min(20, len(search_frame)))
    trailing_20y = search_frame.tail(min(80, len(search_frame)))
    severity = 1.0 if scenario == "adverse" else 2.0

    config_map = {item["key"]: item for item in variable_config["exogenous"] if item["key"] in stressed.columns}
    for column, spec in config_map.items():
        history = search_frame[column].astype(float)
        last_value = float(history.iloc[-1])
        trailing_mean = float(trailing_20q[column].mean())
        trailing_std = float(trailing_20q[column].std(ddof=0))
        if not np.isfinite(trailing_std) or trailing_std <= 0.0:
            trailing_std = max(abs(last_value) * 0.05, 1e-6)
        long_run = float(trailing_20y[column].mean())

        if spec["transform"] == "log_diff":
            stress_target = trailing_mean + severity * trailing_std
            if spec.get("expected_sign") == "positive":
                stress_target = trailing_mean - severity * trailing_std
            if spec.get("expected_sign") == "negative":
                stress_target = trailing_mean + severity * trailing_std
        else:
            stress_target = long_run + severity * trailing_std
            if spec.get("expected_sign") == "positive":
                stress_target = long_run - severity * trailing_std
            if spec.get("expected_sign") == "negative":
                stress_target = long_run + severity * trailing_std

        values: list[float] = []
        for quarter in range(1, horizon + 1):
            if quarter <= 8:
                blend = quarter / 8.0
                value = last_value + (stress_target - last_value) * blend
            elif quarter <= 20:
                value = stress_target
            elif quarter <= 40:
                blend = (quarter - 20) / 20.0
                value = stress_target + (long_run - stress_target) * blend
            else:
                value = long_run
            values.append(float(value))
        stressed[column] = values

    stressed["gfc_dummy"] = 1.0
    stressed["covid_dummy"] = 0.0
    return stressed


def weighted_quantile_mix(frame_a: pd.DataFrame, frame_b: pd.DataFrame, weight_a: float, weight_b: float) -> pd.DataFrame:
    result = pd.DataFrame({"date": frame_a["date"]})
    for column in ["hpi_forecast", "hpi_lower_90", "hpi_upper_90", "hpi_lower_50", "hpi_upper_50"]:
        result[column] = weight_a * frame_a[column].to_numpy(dtype=float) + weight_b * frame_b[column].to_numpy(dtype=float)
    return result


def component_forecast(
    component_name: str,
    spec: dict[str, Any],
    fitted: Any,
    search_frame: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> pd.DataFrame:
    exogenous = list(spec["exogenous"])
    history = search_frame[["date", "hpi"] + exogenous].copy() if exogenous else search_frame[["date", "hpi"]].copy()
    future = future_exog[["date"] + exogenous].copy() if exogenous else future_exog[["date"]].copy()
    predicted = train.tabular_ml_forecast_path(fitted, history, future)
    lower_90, upper_90 = train.regularized_linear_interval(predicted, fitted.residual_std, alpha=0.10)
    lower_50, upper_50 = train.regularized_linear_interval(predicted, fitted.residual_std, alpha=0.50)
    return pd.DataFrame(
        {
            "date": future_exog["date"],
            "component": component_name,
            "hpi_forecast": predicted,
            "hpi_lower_90": lower_90,
            "hpi_upper_90": upper_90,
            "hpi_lower_50": lower_50,
            "hpi_upper_50": upper_50,
        }
    )


def scenario_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "implemented": True,
        "start_forecast": float(frame["hpi_forecast"].iloc[0]),
        "end_forecast": float(frame["hpi_forecast"].iloc[-1]),
        "min_forecast": float(frame["hpi_forecast"].min()),
        "max_forecast": float(frame["hpi_forecast"].max()),
        "q8_forecast": float(frame["hpi_forecast"].iloc[min(7, len(frame) - 1)]),
        "q20_forecast": float(frame["hpi_forecast"].iloc[min(19, len(frame) - 1)]),
        "q40_forecast": float(frame["hpi_forecast"].iloc[min(39, len(frame) - 1)]),
    }


def main() -> int:
    args = parse_args()
    finalized_spec = read_json(Path(args.finalized_spec_path))
    champion_report = read_json(Path(args.champion_report_path))
    search_frame, _, _, variable_config = train.load_panels()
    search_frame = train.filter_sample(search_frame)

    component_specs: dict[str, dict[str, Any]] = {}
    fitted_components: dict[str, Any] = {}
    for component in finalized_spec["components"]:
        spec, fitted = load_component_fit(component, search_frame)
        component_specs[component["component_name"]] = spec
        fitted_components[component["component_name"]] = fitted

    weights = finalized_spec.get("metadata", {}).get("weights", {})
    gb_weight = float(weights["Gradient Boosting__y_pred"])
    xgb_weight = float(weights["XGBoost__y_pred"])

    scenario_outputs: list[pd.DataFrame] = []
    scenario_summaries: dict[str, Any] = {}
    for scenario in ["baseline", "adverse", "severely_adverse"]:
        future_exog = stressed_future_exog(search_frame, variable_config, horizon=120, scenario=scenario)
        gb_forecast = component_forecast(
            "Gradient Boosting",
            component_specs["Gradient Boosting"],
            fitted_components["Gradient Boosting"],
            search_frame,
            future_exog,
        )
        xgb_forecast = component_forecast(
            "XGBoost",
            component_specs["XGBoost"],
            fitted_components["XGBoost"],
            search_frame,
            future_exog,
        )
        ensemble = weighted_quantile_mix(gb_forecast, xgb_forecast, gb_weight, xgb_weight)
        ensemble.insert(1, "scenario", scenario)
        scenario_outputs.append(ensemble)
        scenario_summaries[scenario] = scenario_summary(ensemble)

    combined = pd.concat(scenario_outputs, ignore_index=True)
    combined.to_csv(Path(args.output_path), index=False)

    champion_report["scenario_summary"] = scenario_summaries
    champion_report["scenario_output_path"] = str(Path(args.output_path))
    write_json(Path(args.champion_report_path), champion_report)
    print(f"Wrote {args.output_path}")
    print(f"Updated {args.champion_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
