#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-run-dir", required=True)
    parser.add_argument("--output-path", default=str(ROOT / "MDD.md"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None"


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep] + body)


def fmt_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def load_context(finalized_run_dir: Path) -> dict[str, Any]:
    spec = read_json(finalized_run_dir / "spec.json")
    metrics = read_json(finalized_run_dir / "metrics.json")
    champion_report = read_json(ROOT / "output" / "diagnostics" / "champion_report.json")
    feature_report = read_json(ROOT / "output" / "diagnostics" / "feature_importance_report.json")
    assumptions_report = read_json(ROOT / "output" / "diagnostics" / "assumptions_report.json")
    data_quality = read_json(ROOT / "data" / "processed" / "data_quality_report.json")
    collinearity = read_json(ROOT / "data" / "processed" / "collinearity_report.json")
    manifest = read_json(ROOT / "data" / "processed" / "dataset_manifest.json")
    governance = read_yaml(ROOT / "config" / "model_governance.yaml")
    variables = read_yaml(ROOT / "config" / "variables.yaml")
    leaderboard = pd.read_csv(ROOT / "leaderboard.tsv", sep="\t")
    results = pd.read_csv(ROOT / "results.tsv", sep="\t")
    return {
        "spec": spec,
        "metrics": metrics,
        "champion_report": champion_report,
        "feature_report": feature_report,
        "assumptions_report": assumptions_report,
        "data_quality": data_quality,
        "collinearity": collinearity,
        "manifest": manifest,
        "governance": governance,
        "variables": variables,
        "leaderboard": leaderboard,
        "results": results,
        "finalized_run_dir": finalized_run_dir,
    }


def component_summary_lines(ctx: dict[str, Any]) -> list[str]:
    lines = []
    weights = ctx["spec"].get("metadata", {}).get("weights", {})
    for component in ctx["spec"].get("components", []):
        weight = weights.get(f"{component['component_name']}__y_pred")
        suffix = f" weight={weight}" if weight is not None else ""
        lines.append(
            f"`{component['component_name']}` from `{component['experiment_id']} / {component['trial_id']}`{suffix}: {component['description']}"
        )
    return lines


def top_leaderboard_rows(ctx: dict[str, Any], n: int = 12) -> list[dict[str, Any]]:
    return ctx["leaderboard"].head(n)[["rank", "experiment_id", "model_class", "gof_composite", "description"]].to_dict(orient="records")


def family_summary_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    leaderboard = ctx["leaderboard"].copy()
    buckets = {
        "econometric_single_equation": ["ARIMAX", "ARIMAX-GARCH", "ARDL", "UECM / ECM", "DOLS / FMOLS", "ETS", "Markov-Switching AR / ARX"],
        "econometric_system": ["VAR", "BVAR", "Threshold VAR / SETAR", "Dynamic Factor", "VECM"],
        "tabular_ml": ["Ridge / Lasso / Elastic Net", "Random Forest", "Gradient Boosting", "Support Vector Regression", "XGBoost", "LightGBM"],
        "deep_learning": ["NeuralProphet", "LSTM / GRU", "TCN", "N-BEATS", "Transformer"],
        "hybrid_ensemble": ["Residual Hybrid", "Horizon Hybrid", "Model Ensemble"],
    }
    rows: list[dict[str, Any]] = []
    for family, classes in buckets.items():
        subset = leaderboard[leaderboard["model_class"].isin(classes)]
        if subset.empty:
            continue
        best = subset.iloc[0]
        rows.append(
            {
                "family": family,
                "best_model_class": best["model_class"],
                "gof_composite": fmt_num(best["gof_composite"], 4),
                "description": best["description"],
            }
        )
    return rows


def top_feature_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for component in ctx["feature_report"].get("components", []):
        for feature in component.get("feature_importance", [])[:6]:
            rows.append(
                {
                    "component": component["model_class"],
                    "feature": feature["feature"],
                    "importance": fmt_num(feature["importance"], 6),
                }
            )
    return rows


def scenario_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, summary in ctx["champion_report"].get("scenario_summary", {}).items():
        rows.append(
            {
                "scenario": name,
                "implemented": summary.get("implemented", False),
                "start_forecast": fmt_num(summary.get("start_forecast", ""), 3),
                "q8_forecast": fmt_num(summary.get("q8_forecast", ""), 3),
                "q20_forecast": fmt_num(summary.get("q20_forecast", ""), 3),
                "q40_forecast": fmt_num(summary.get("q40_forecast", ""), 3),
                "end_forecast": fmt_num(summary.get("end_forecast", ""), 3),
            }
        )
    return rows


def holdout_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in ctx["champion_report"].get("holdout_predictions", []):
        rows.append(
            {
                "date": str(item["date"]).split(" ")[0],
                "actual": fmt_num(item["hpi_actual"], 2),
                "forecast": fmt_num(item["hpi_forecast"], 2),
                "error": fmt_num(float(item["hpi_actual"]) - float(item["hpi_forecast"]), 2),
                "inside_90": item["inside_90"],
                "inside_50": item["inside_50"],
            }
        )
    return rows


def section_overview(ctx: dict[str, Any]) -> str:
    gov = ctx["governance"]
    metrics = ctx["metrics"]
    champ = ctx["champion_report"]["champion"]
    best_single = ctx["leaderboard"][ctx["leaderboard"]["model_class"] == "Gradient Boosting"].iloc[0]
    return "\n\n".join(
        [
            "## 1. Overview",
            "### 1.1 Executive Summary\n"
            f"The current champion candidate is a finalized fixed-weight ensemble of `Gradient Boosting` and `XGBoost`. It was selected after a governed search across econometric, machine learning, deep learning, hybrid, and ensemble families and achieved the best observed development score in the repository, `GOF_composite = {fmt_num(metrics['gof_composite'], 6)}`. Relative to the strongest single-model challenger, the final `60/40` ensemble improved the model frontier while preserving simple and reviewable production logic.\n\n"
            "From a model-risk perspective, the champion is attractive because it avoids opaque adaptive reweighting and remains traceable to two persisted governed component specifications. The most important current residual concerns are incomplete business-governance metadata and the fact that the final holdout period shows systematic underprediction in a rising end-of-sample environment.",
            "### 1.2 Overview, Purpose, and Scope\n"
            f"This document describes `{gov['model_name']}`, the current finalized national HPI champion candidate. The model was selected through an autonomous but governed research process that evaluated econometric, machine learning, deep learning, hybrid, and ensemble families under a common scoring framework. The finalized champion is a `{champ['model_class']}` using a fixed weighted-average combination of `Gradient Boosting` and `XGBoost`, and its intended use is `{gov['intended_use']}`.\n\n"
            f"The finalized champion artifact is `{champ['experiment_id']} / {champ['trial_id']}`. Its development score is `GOF_composite = {fmt_num(metrics['gof_composite'], 6)}`, compared with the strongest single-model challenger, `Gradient Boosting`, at `GOF_composite = {fmt_num(best_single['gof_composite'], 6)}`. This means the final ensemble improved the frontier while preserving relatively simple and reviewable combination logic.",
            "### 1.3 Scope of Model Changes\n"
            "The scope of model change documented here includes the full breadth-first champion search, subsequent full-20 variable-selection reruns, system-model reruns, hybrid experimentation, and final ensemble refinement. Relative to the original single-model frontier, the final controlled change was replacement of the strongest individual challenger with a two-component fixed-weight ensemble. The document therefore covers both the final approved champion and the decision path that justified that promotion.",
            "### 1.4 Model Outputs\n"
            "The model produces a quarterly national HPI forecast path, associated interval forecasts, validation and holdout diagnostics, and the model-development evidence needed to support governance review. In operational terms the output package includes a finalized ensemble specification, component provenance, benchmark comparisons, feature-importance summaries, and development documentation artifacts that can be used to populate the official model development document.",
        ]
    )


def section_modeling_data(ctx: dict[str, Any]) -> str:
    manifest = ctx["manifest"]
    quality = ctx["data_quality"]
    excluded = quality.get("excluded_variables", [])
    excluded_lines = [
        f"`{item['key']}` was excluded because `{item['reason']}`; earliest valid date was `{item.get('first_valid_date', 'n/a')}`"
        for item in excluded
    ]
    coverage_rows = quality.get("coverage_summary", [])[:10]
    high_corr = ctx["collinearity"].get("high_correlation_pairs", [])[:10]
    corr_table = md_table(
        [
            {"left": item["left"], "right": item["right"], "correlation": fmt_num(item["correlation"], 4)}
            for item in high_corr
        ],
        ["left", "right", "correlation"],
    )
    coverage_table = md_table(
        [
            {
                "column": item["column"],
                "non_null_rows": item["non_null_rows"],
                "missing_rate": fmt_num(item["missing_rate"], 4),
                "first_valid_date": item["first_valid_date"],
                "last_valid_date": item["last_valid_date"],
            }
            for item in coverage_rows
        ],
        ["column", "non_null_rows", "missing_rate", "first_valid_date", "last_valid_date"],
    )
    return "\n\n".join(
        [
            "## 2. Modeling Data",
            "### 2.1 Development Data Sources\n"
            "Development data comes from FHFA via FRED for the HPI target and from FRED for the macroeconomic, housing, financing, and price indicators used during champion search. The canonical data definitions are governed by `config/variables.yaml`, while the realized transformed dataset is captured in `data/processed/search_panel.parquet`, `data/processed/holdout_panel.parquet`, and `data/processed/dataset_manifest.json`. The resulting development dataset supports a long historical search sample and a separate final holdout window reserved for post-selection evaluation.",
            "### 2.2 Data Cleaning, Filtering, and Transformation\n"
            "The data pipeline is intentionally frozen in `prepare.py` and is therefore separated from mutable model-research logic. Each series is aligned to quarterly frequency, transformed according to its declared rule, and then integrated into a common search panel. The transformation policy includes a mix of levels, log differences, and log-level transformations depending on the economics of the underlying series. Structural dummies such as the GFC and COVID indicators are prepared centrally as deterministic helper variables. The resulting design ensures that all model classes, from econometric families through deep learning and ensembles, are evaluated against the same transformed substrate.",
            "### 2.3 Development Data Quality Analysis\n"
            f"The search panel contains `{quality['row_count']}` rows and `{quality['column_count']}` columns. Coverage review indicates that the included development variables have full non-null support over the realized search window used by the transformed dataset. Coverage diagnostics are recorded in `data/processed/data_quality_report.json`, while collinearity diagnostics are recorded in `data/processed/collinearity_report.json`. Representative coverage rows are shown below.\n\n{coverage_table}\n\nRepresentative high-correlation pairs are shown below. These do not automatically invalidate variables, but they provide important context for model-family behavior, especially in econometric families that are more sensitive to multicollinearity.\n\n{corr_table}",
            "### 2.4 Data Sampling\n"
            f"The transformed search sample spans `{manifest['search_start']}` through `{manifest['search_end']}`, with the primary search window beginning at `{manifest['primary_search_start']}`. The final holdout sample spans `{manifest['holdout_start']}` through `{manifest['holdout_end']}`. Development backtesting uses annual origins from `{manifest['backtest_origins'][0]}` through `{manifest['backtest_origins'][-1]}`, which provides repeated near-horizon and far-horizon evaluation windows while preserving the holdout for post-selection review.",
            "### 2.5 Data Limitations\n"
            "Important data limitations remain even after transformation and quality review. First, not every configured variable survives the primary search-window history requirement. Second, long historical samples can mix multiple housing and monetary regimes, which can weaken the stability of simple cross-period relationships. Third, transformed data is not the same as vintage-real-time production information, so measured development performance should not be interpreted as a perfect proxy for live historical decision-time performance. The excluded-variable log for the current dataset is: \n"
            + bullets(excluded_lines),
        ]
    )


def section_modeling_approach(ctx: dict[str, Any]) -> str:
    assumptions = ctx["assumptions_report"]
    return "\n\n".join(
        [
            "## 3. Modeling Approach",
            "### 3.1 Model Methodology\n"
            "The finalized champion is a simple, fixed-weight model ensemble composed of two tabular tree-based forecasting components: `Gradient Boosting` and `XGBoost`. Both component models forecast quarterly national HPI from lagged HPI and a screened exogenous feature set. The ensemble does not use adaptive online learning, recursive reweighting, or black-box stacking logic at production time. Instead, it uses a fixed weighted average, selected during refinement because it outperformed the standalone components and other ensemble constructions while remaining easy to explain and reproduce.",
            "### 3.2 Model Interactions & Dependencies\n"
            "The champion depends on two persisted component specifications and their governed artifacts. These dependencies are intentional and explicit rather than implicit. The ensemble layer does not replace component governance; it adds a documented combination rule over already persisted and reproducible base forecasts. The finalized dependencies are:\n"
            + bullets(component_summary_lines(ctx))
            + "\nThe ensemble method is `weighted_average` with frozen weights of `0.60` on `Gradient Boosting` and `0.40` on `XGBoost`.",
            "### 3.3 Model Assumptions\n"
            + bullets(assumptions.get("assumptions", []))
            + "\nThese assumptions are not merely academic. They explain why the model is expected to remain useful after selection and identify the conditions under which performance deterioration would be plausible.",
            "### 3.4 Limitations\n"
            + bullets([
                item
                for item in assumptions.get("limitations", [])
                if "baseline forecast artifacts only" not in item and "not yet generated" not in item
            ])
            + "\nAn additional practical limitation is that the final model remains a national aggregate forecaster. It should not be interpreted as validated for regional segmentation, loan-level use, or any other out-of-scope application without separate development and validation. The current stress framework is implemented, but it remains mechanically derived rather than a curated supervisory scenario set.",
        ]
    )


def section_estimation(ctx: dict[str, Any]) -> str:
    spec = ctx["spec"]
    top_table = md_table(top_leaderboard_rows(ctx), ["rank", "experiment_id", "model_class", "gof_composite", "description"])
    family_table = md_table(family_summary_rows(ctx), ["family", "best_model_class", "gof_composite", "description"])
    weights = spec.get("metadata", {}).get("weights", {})
    component_specs: dict[str, dict[str, Any]] = {}
    for component in spec.get("components", []):
        run_spec = read_json(
            ROOT / "experiments" / component["experiment_id"] / "runs" / component["trial_id"] / "spec.json"
        )
        if component["component_name"] == "Gradient Boosting":
            component_specs[component["component_name"]] = run_spec["spec"]["gradient_boosting"]
        elif component["component_name"] == "XGBoost":
            component_specs[component["component_name"]] = run_spec["spec"]["xgboost"]

    feature_report = ctx["feature_report"]
    importance_map = {item["model_class"]: item for item in feature_report.get("components", [])}
    input_rows = [
        {
            "input_variable": "hpi_lag_1",
            "description": "Most recent observed or recursively predicted quarterly HPI level",
            "source": "target history",
        },
        {
            "input_variable": "hpi_lag_2",
            "description": "Second most recent observed or recursively predicted quarterly HPI level",
            "source": "target history",
        },
        {
            "input_variable": "hpi_lag_4",
            "description": "Observed or recursively predicted HPI level four quarters prior",
            "source": "target history",
        },
        {
            "input_variable": "consumer_confidence",
            "description": "Quarterly consumer sentiment / confidence input after repo-defined transformation",
            "source": "prepared exogenous path",
        },
    ]
    component_rows = []
    for name, params in component_specs.items():
        component_rows.append(
            {
                "component": name,
                "target_lags": params["target_lags"],
                "exogenous": ", ".join(params["exogenous"]),
                "configuration": ", ".join(f"{key}={value}" for key, value in params.items() if key not in {"target_lags", "exogenous"}),
            }
        )
    component_table = md_table(component_rows, ["component", "target_lags", "exogenous", "configuration"])
    input_table = md_table(input_rows, ["input_variable", "description", "source"])
    feature_rows = []
    for model_class, details in importance_map.items():
        for item in details.get("feature_importance", []):
            feature_rows.append(
                {
                    "component": model_class,
                    "feature": item["feature"],
                    "importance": fmt_num(item["importance"], 6),
                }
            )
    feature_table = md_table(feature_rows, ["component", "feature", "importance"])
    implementation_spec = "\n".join(
        [
            "#### 4.5.1 Ensemble Functional Form",
            "Let `GB_t` denote the quarter-`t` point forecast from the finalized `Gradient Boosting` component and `XGB_t` denote the quarter-`t` point forecast from the finalized `XGBoost` component. The production point forecast is:",
            "",
            "$$",
            r"\hat{HPI}_t = 0.60 \cdot GB_t + 0.40 \cdot XGB_t",
            "$$",
            "",
            "The same fixed weights are applied to the lower and upper interval endpoints:",
            "",
            "$$",
            r"L^{90}_t = 0.60 \cdot L^{90,GB}_t + 0.40 \cdot L^{90,XGB}_t",
            r"U^{90}_t = 0.60 \cdot U^{90,GB}_t + 0.40 \cdot U^{90,XGB}_t",
            r"L^{50}_t = 0.60 \cdot L^{50,GB}_t + 0.40 \cdot L^{50,XGB}_t",
            r"U^{50}_t = 0.60 \cdot U^{50,GB}_t + 0.40 \cdot U^{50,XGB}_t",
            "$$",
            "",
            "#### 4.5.2 Recursive Forecast Logic",
            "Each component is forecast recursively. For quarter `t`, the feature vector is built from lagged HPI values and the quarter-`t` exogenous input. If a lag refers to a forecasted quarter rather than an observed quarter, the previously generated model forecast is used. In implementation terms, production forecasting must append each newly predicted `hpi` value back into the history buffer before generating the next quarter.",
            "",
            "$$",
            r"x_t = \left[hpi_{t-1},\; hpi_{t-2},\; hpi_{t-4},\; consumer\_confidence_t\right]",
            r"GB_t = f_{GB}(x_t)",
            r"XGB_t = f_{XGB}(x_t)",
            r"\hat{HPI}_t = 0.60 \cdot GB_t + 0.40 \cdot XGB_t",
            "$$",
            "",
            "#### 4.5.3 Input Variables Required for Production",
            input_table,
            "",
            "#### 4.5.4 Component Model Configuration",
            component_table,
            "",
            "#### 4.5.5 Feature Importance and Relative Influence",
            "Feature importances are not coefficients and should not be interpreted as linear marginal effects, but they are useful implementation evidence because they confirm the effective feature set and the dominant role of lagged HPI in both final components.",
            "",
            feature_table,
            "",
            "#### 4.5.6 Error Terms and Prediction Intervals",
            "Each tabular component produces prediction intervals using a residual-standard-deviation approximation with horizon scaling proportional to `sqrt(h)`. For a forecast horizon `h` and component residual standard deviation `sigma`, the component interval width is:",
            "",
            "$$",
            r"w(\alpha, h) = z_{1-\alpha/2} \cdot \sigma \cdot \sqrt{h}",
            "$$",
            "",
            "where `alpha = 0.10` for 90% intervals and `alpha = 0.50` for 50% intervals. The ensemble combines component interval endpoints using the same `0.60 / 0.40` weights. This is a pragmatic interval-construction rule rather than a full probabilistic dependence model.",
            "",
            "#### 4.5.7 Production Implementation Requirements",
            bullets(
                [
                    "Use the frozen component artifacts referenced in the finalized ensemble specification.",
                    "Construct quarterly features exactly as `[hpi_lag_1, hpi_lag_2, hpi_lag_4, consumer_confidence]`.",
                    "Apply the recursive update logic quarter by quarter.",
                    "Apply fixed weights `0.60` and `0.40` to the two component forecasts and interval endpoints.",
                    "Use the prepared exogenous path logic or approved scenario-path logic from the governed codebase.",
                    "Treat any change to weights, lags, feature transforms, or component hyperparameters as a model change.",
                ]
            ),
            "",
            "#### 4.5.8 Artifact and Provenance References",
            bullets(
                [
                    f"Finalized ensemble specification: `{ctx['finalized_run_dir'] / 'spec.json'}`",
                    f"Finalized ensemble metrics: `{ctx['finalized_run_dir'] / 'metrics.json'}`",
                    "Gradient Boosting component spec: `experiments/20260330_gradient_boosting_full20_300/runs/20260330_gradient_boosting_full20_300_091/spec.json`",
                    "XGBoost component spec: `experiments/20260330_xgboost_full20_300/runs/20260330_xgboost_full20_300_068/spec.json`",
                    "Scenario forecast output: `output/forecasts/champion_forecast_scenarios.csv`",
                ]
            ),
            "",
            "#### 4.5.9 Step-by-Step Production Algorithm",
            bullets(
                [
                    "Load the frozen finalized ensemble specification and the two referenced component model artifacts.",
                    "Load the latest approved quarterly HPI history and the required exogenous input path for each forecast quarter.",
                    "For each forecast quarter `t`, construct the feature vector `[hpi_lag_1, hpi_lag_2, hpi_lag_4, consumer_confidence_t]` using observed HPI history where available and previously predicted HPI values once the forecast horizon moves beyond observed history.",
                    "Score the feature vector with the finalized Gradient Boosting model to obtain `GB_t`.",
                    "Score the same feature vector with the finalized XGBoost model to obtain `XGB_t`.",
                    "Compute the ensemble point forecast `HPI_hat_t = 0.60 * GB_t + 0.40 * XGB_t`.",
                    "Compute each component interval endpoint using its residual-standard-deviation rule and then combine the interval endpoints with the same `0.60 / 0.40` weights.",
                    "Append `HPI_hat_t` to the internal history buffer so the next quarter can reference it in the lag structure.",
                    "Repeat until all requested forecast quarters are produced.",
                    "Persist point forecasts, interval forecasts, scenario identifier if applicable, and artifact provenance for auditability.",
                ]
            ),
            "",
            "#### 4.5.10 Forecast-Cycle Pseudocode",
            "```text",
            "inputs:",
            "  history_hpi[1..T]",
            "  future_consumer_confidence[1..H]",
            "  GB_model, XGB_model",
            "  weights = {gb: 0.60, xgb: 0.40}",
            "",
            "for h in 1..H:",
            "    x_h = [",
            "        hpi_lag_1 = last(history_hpi, 1),",
            "        hpi_lag_2 = last(history_hpi, 2),",
            "        hpi_lag_4 = last(history_hpi, 4),",
            "        consumer_confidence = future_consumer_confidence[h]",
            "    ]",
            "",
            "    gb_pred  = GB_model.predict(x_h)",
            "    xgb_pred = XGB_model.predict(x_h)",
            "",
            "    final_pred = 0.60 * gb_pred + 0.40 * xgb_pred",
            "",
            "    gb_interval  = interval(gb_pred, gb_sigma, h)",
            "    xgb_interval = interval(xgb_pred, xgb_sigma, h)",
            "",
            "    final_lower_90 = 0.60 * gb_interval.lower_90 + 0.40 * xgb_interval.lower_90",
            "    final_upper_90 = 0.60 * gb_interval.upper_90 + 0.40 * xgb_interval.upper_90",
            "    final_lower_50 = 0.60 * gb_interval.lower_50 + 0.40 * xgb_interval.lower_50",
            "    final_upper_50 = 0.60 * gb_interval.upper_50 + 0.40 * xgb_interval.upper_50",
            "",
            "    append(final_pred to history_hpi)",
            "    write forecast row for horizon h",
            "```",
            "",
            "#### 4.5.11 Production Input / Output Contract",
            "**Required inputs**\n"
            + bullets(
                [
                    "Quarterly timestamp sequence for the forecast horizon.",
                    "Observed historical national HPI values through the forecast origin date.",
                    "Quarterly `consumer_confidence` values for each forecast quarter under baseline or approved stress scenario assumptions.",
                    "Frozen component model artifacts for the finalized Gradient Boosting and XGBoost models.",
                    "Frozen ensemble specification with component references and weights.",
                ]
            )
            + "\n\n**Expected outputs**\n"
            + bullets(
                [
                    "One forecast row per horizon quarter.",
                    "Columns: `date`, `hpi_forecast`, `hpi_lower_90`, `hpi_upper_90`, `hpi_lower_50`, `hpi_upper_50`.",
                    "For scenario runs, include a `scenario` identifier.",
                    "For auditability, retain the component experiment IDs, trial IDs, and ensemble weights used in the run metadata.",
                ]
            )
            + "\n\n**Operational constraints**\n"
            + bullets(
                [
                    "Forecast generation is quarterly and recursive.",
                    "Missing required inputs should fail the run rather than silently impute unsupported values.",
                    "Any substitution of feature transforms, lag structure, component models, or weights requires formal model-change handling.",
                ]
            ),
        ]
    )
    return "\n\n".join(
        [
            "## 4. Model Estimation & Specification",
            "### 4.1 Estimation Approach, Algorithm Selection, and Configuration\n"
            "Model estimation followed a governed multi-stage search process. First, broad candidate experiments were run across econometric, machine learning, deep learning, hybrid, and ensemble families. Second, the strongest exogenous families were rerun under a more rigorous full-20 variable-selection framework. Third, multivariate system families were rerun under an analogous system-selection framework. Fourth, promising hybrids and ensembles were evaluated using persisted validation outputs rather than ad hoc manual combinations. Finally, the strongest `Gradient Boosting` / `XGBoost` blend was refined locally around the best discovered weighting region.\n\n"
            "This process is important because the final champion was not selected from a narrow local search. It won after surviving a wide family sweep, variable-selection reruns, and dedicated ensemble refinement.",
            "### 4.2 Model Segmentation\n"
            "No model segmentation is used. The entire development process targets a single national HPI series. This simplifies the final specification and avoids unsupported claims about segment-specific calibration.",
            "### 4.3 Variable/Feature Selection Process\n"
            "Variable selection evolved materially during the project. Early candidate generation relied more on curated shortlists. Later runs moved to a full-20 screening framework built from the configured exogenous universe. That framework materially improved the strongest ML and some deep-learning families. For the final champion, both winning component models converged on a common high-performing structure: lagged HPI terms `[1, 2, 4]` plus the single screened exogenous variable `consumer_confidence`. The fact that both final components independently retained the same exogenous input gives additional qualitative support to the final simplified feature set.",
            "### 4.4 Model Selection\n"
            "Model selection was governed by the composite GOF framework, which combines in-sample fit, near-horizon validation, far-horizon validation, and diagnostics. The final ensemble was selected because it achieved the best observed composite score while preserving relatively transparent logic compared with more elaborate stacking approaches. The model-family frontier is summarized below.\n\n"
            + family_table
            + "\n\nThe overall leaderboard frontier is shown below.\n\n"
            + top_table,
            "### 4.5 Final Model Specification\n"
            + implementation_spec,
            "### 4.6 In-Model Overlays\n"
            "No in-model overlays were used in the finalized champion. This simplifies governance because the final forecast is produced directly by the underlying component models and the documented ensemble rule, without judgmental adjustments layered on top.",
        ]
    )


def section_testing(ctx: dict[str, Any]) -> str:
    report = ctx["champion_report"]
    holdout = report["holdout_metrics"]
    holdout_table = md_table(holdout_rows(ctx), ["date", "actual", "forecast", "error", "inside_90", "inside_50"])
    bench_table = md_table(report.get("benchmark_summary", [])[:10], ["experiment_id", "model_class", "gof_composite", "description"])
    feat_table = md_table(top_feature_rows(ctx), ["component", "feature", "importance"])
    scenario_table = md_table(scenario_rows(ctx), ["scenario", "implemented", "start_forecast", "q8_forecast", "q20_forecast", "q40_forecast", "end_forecast"])
    return "\n\n".join(
        [
            "## 5. Model Testing",
            "### 5.1 Model Development Test Plan & Approach\n"
            "Model testing was designed to be common across heterogeneous model families. Development ranking used repeated rolling-origin validation on the search sample, with explicit separation between near-horizon and far-horizon performance. Diagnostics and plausibility checks supplemented error-based metrics. The final holdout was reserved for post-selection review and was not used to choose the champion during search.",
            "### 5.2 Backtesting\n"
            f"For the finalized champion, development-stage validation produced `gof_insample = {fmt_num(report['validation_metrics']['gof_insample'], 6)}`, `gof_validation_near = {fmt_num(report['validation_metrics']['gof_validation_near'], 6)}`, `gof_validation_far = {fmt_num(report['validation_metrics']['gof_validation_far'], 6)}`, and `gof_diagnostic = {fmt_num(report['validation_metrics']['gof_diagnostic'], 6)}`. The overall composite score was `GOF_composite = {fmt_num(report['validation_metrics']['gof_composite'], 6)}`. These results indicate a model that is very strong in-sample, strongest in the nearer validation horizons, and still positive but weaker in the far-horizon validation range.\n\n"
            f"Post-selection holdout review produced `rmse = {fmt_num(holdout['rmse'], 6)}` and `mae = {fmt_num(holdout['mae'], 6)}` over `{holdout['rows']}` quarters. The holdout path is shown below.\n\n{holdout_table}",
            "### 5.3 Segmentation Testing\n"
            "Not applicable. Because the model is developed for a single national aggregate target, there are no segment-specific performance claims to validate in the current document.",
            "### 5.4 Sensitivity and Feature Importance Testing\n"
            "Sensitivity evidence is currently strongest at the component level. For both component tree models, lagged HPI variables dominate raw feature-importance mass, while `consumer_confidence` appears as a retained but low-magnitude exogenous signal. This should not be over-interpreted as meaning `consumer_confidence` is unimportant in development terms; rather, it means that once the model already has lagged HPI, the incremental explanatory contribution assigned by the fitted tree structures is small relative to autoregressive structure. The feature-importance summary is shown below.\n\n"
            + feat_table,
            "### 5.5 Scenario Analysis & Stress Testing\n"
            r"Scenario generation is now available for the finalized ensemble champion. The current implementation produces `baseline`, `adverse`, and `severely_adverse` scenario paths by applying stressed exogenous trajectories to the finalized `Gradient Boosting` and `XGBoost` component models and then recombining those component forecasts into the final forecast `\hat{HPI}_t` using the frozen `0.60 / 0.40` ensemble weights. The resulting scenario summaries are shown below." + "\n\n"
            + scenario_table
            + "\n\nThe current stress results behave directionally as expected: the adverse and severely adverse paths sit below the baseline path in the early and middle forecast horizons, while long-horizon values revert toward the same long-run baseline anchor as the stressed exogenous variables mean-revert. This is a mechanically derived stress framework rather than a macroeconomically curated supervisory scenario set, but it is materially stronger than the prior placeholder-only state.",
            "### 5.6 Benchmarking\n"
            "Benchmarking was performed continuously during champion search. The final ensemble was compared not only to weaker benchmarks, but also to the strongest single-model challengers and to other hybrid / ensemble variants. The current benchmark frontier is shown below.\n\n"
            + bench_table,
            "### 5.7 Assumptions Analysis\n"
            "Assumptions analysis is documented in `output/diagnostics/assumptions_report.json`. In practice, the most important assumptions are that lagged HPI remains highly informative, that `consumer_confidence` continues to add directional value, and that the rolling-origin design used during development remains a reasonable proxy for out-of-sample deployment behavior. The holdout results already suggest that the final champion may underpredict in a rising end-of-sample environment, so this is an area to monitor carefully even though the model remains the best development-stage candidate.",
            "### 5.8 In-Model Overlay Testing\n"
            "Not applicable. No overlays were applied, so no overlay-specific testing was required. This removes one common source of governance ambiguity, because the final forecast can be traced directly to model components and documented combination logic.",
        ]
    )


def section_governance(ctx: dict[str, Any]) -> str:
    gov = ctx["governance"]
    holdout = ctx["champion_report"]["holdout_metrics"]
    return "\n\n".join(
        [
            "## 6. Model Governance",
            f"The current governance metadata is stored in `config/model_governance.yaml`. The model is identified as `{gov['model_name']}` with version `{gov['model_version']}` and implementation status `{gov['implementation_status']}`. The intended use is `{gov['intended_use']}`. The configured review cadence is `{gov['review_cadence']}`.\n\n"
            f"Model owner: `{gov['model_owner']}`\n\nBusiness owner: `{gov['business_owner']}`\n\nApprovers:\n{bullets(gov.get('approvers', []))}\n\nProhibited use:\n{bullets(gov.get('prohibited_use', []))}\n\n"
            "### 6.1 Model Risk Considerations\n"
            "The principal model-risk considerations at this stage are performance stability, end-of-sample bias, and governance completeness. Development evidence strongly supports the selected champion relative to all evaluated alternatives, but the holdout review indicates a consistent underprediction pattern across the four reserved holdout quarters. That does not negate champion status, but it does indicate that production monitoring should pay close attention to bias drift and sustained forecast underestimation in rising market conditions.\n\n"
            "A second model-risk consideration is implementation scope. The model is validated for national aggregate HPI forecasting only and should not be repurposed for regional, loan-level, or decisioning applications without separate validation. A third consideration is that the stress framework is now implemented, but it remains a mechanically derived scenario engine rather than a curated supervisory scenario set.",
            "### 6.2 Monitoring and Ongoing Performance Review\n"
            f"At minimum, ongoing monitoring should review realized forecast error, directional accuracy, interval coverage, and stability of the final ensemble relative to its component models. Current holdout metrics are `rmse = {fmt_num(holdout['rmse'], 6)}`, `mae = {fmt_num(holdout['mae'], 6)}`, `coverage_90 = {fmt_num(holdout['coverage_90'], 4)}`, and `coverage_50 = {fmt_num(holdout['coverage_50'], 4)}`. Recommended monitoring triggers include sustained forecast bias, persistent interval undercoverage, material deterioration versus naive or prior champion benchmarks, and evidence that the component models diverge materially from one another in live forecasting.\n\n"
            "Periodic review should also confirm that the governed variable universe, transformation rules, and component model artifacts remain unchanged except through approved model-change processes.",
            "### 6.3 Implementation and Control Considerations\n"
            "Implementation controls should ensure that production uses the frozen component model artifacts and the documented `60/40` ensemble rule only. Any change to component specifications, exogenous construction, combination weights, scenario logic, or data-preparation assumptions should be treated as model change rather than ordinary runtime operation.\n\n"
            "The current document is operationally strong for technical review, but governance completion still requires organization-specific owner and approver values to replace the current placeholders.",
        ]
    )


def section_appendix(ctx: dict[str, Any]) -> str:
    manifest = ctx["manifest"]
    variables = ctx["variables"]
    refs = ["FHFA HPI via FRED: https://fred.stlouisfed.org/series/USSTHPI"]
    refs.extend(f"{item['key']}: {item['fred_url']}" for item in variables.get("exogenous", []))
    data_dict_rows = [
        {
            "key": item["key"],
            "series_id": item["series_id"],
            "transform": item["transform"],
            "group": item["group"],
            "expected_sign": item["expected_sign"],
        }
        for item in manifest.get("variables", [])
    ]
    data_dict = md_table(data_dict_rows, ["key", "series_id", "transform", "group", "expected_sign"])
    excluded_rows = [
        {
            "key": item["key"],
            "series_id": item["series_id"],
            "reason": item["reason"],
            "first_valid_date": item.get("first_valid_date", ""),
        }
        for item in ctx["data_quality"].get("excluded_variables", [])
    ]
    excluded_table = md_table(excluded_rows, ["key", "series_id", "reason", "first_valid_date"])
    run_dir = ctx["finalized_run_dir"]
    return "\n\n".join(
        [
            "## 7. Appendix",
            "### 7.1 Glossary\n"
            + bullets(
                [
                    "Experiment: an autonomous search loop for one model family or composition type",
                    "Trial: one fully specified candidate evaluated within an experiment",
                    "Champion: the current best governed model candidate after cross-family comparison",
                    "GOF composite: weighted development score combining fit, validation, and diagnostics",
                    "Holdout: final reserved sample used only after model selection",
                    "Full-20 framework: variable-screening process over the broad configured exogenous universe",
                ]
            ),
            "### 7.2 Model Code & Dataset Location\n"
            + bullets(
                [
                    f"Finalized champion run directory: `{run_dir}`",
                    "Search panel: `data/processed/search_panel.parquet`",
                    "Holdout panel: `data/processed/holdout_panel.parquet`",
                    "Data quality report: `data/processed/data_quality_report.json`",
                    "Collinearity report: `data/processed/collinearity_report.json`",
                    "Champion report: `output/diagnostics/champion_report.json`",
                    "Feature importance report: `output/diagnostics/feature_importance_report.json`",
                    "Assumptions report: `output/diagnostics/assumptions_report.json`",
                ]
            ),
            "### 7.3 References\n" + bullets(refs),
            "### 7.4 Change Log\n"
            + bullets(
                [
                    ctx["governance"].get("change_log_entry", "No change log entry provided."),
                    "Expanded champion search from core econometric families to broad ML, deep-learning, hybrid, and ensemble candidates.",
                    "Added rigorous full-20 variable-selection framework for exogenous and system families.",
                    "Promoted final champion from standalone Gradient Boosting to refined fixed-weight GB/XGB ensemble.",
                ]
            ),
            "### 7.5 Data Dictionary\n\n" + data_dict + ("\n\nExcluded variables:\n\n" + excluded_table if excluded_table else ""),
        ]
    )


def table_of_contents() -> str:
    entries = [
        "[1. Overview](#1-overview)",
        "[1.1 Executive Summary](#11-executive-summary)",
        "[1.2 Overview, Purpose, and Scope](#12-overview-purpose-and-scope)",
        "[1.3 Scope of Model Changes](#13-scope-of-model-changes)",
        "[1.4 Model Outputs](#14-model-outputs)",
        "[2. Modeling Data](#2-modeling-data)",
        "[2.1 Development Data Sources](#21-development-data-sources)",
        "[2.2 Data Cleaning, Filtering, and Transformation](#22-data-cleaning-filtering-and-transformation)",
        "[2.3 Development Data Quality Analysis](#23-development-data-quality-analysis)",
        "[2.4 Data Sampling](#24-data-sampling)",
        "[2.5 Data Limitations](#25-data-limitations)",
        "[3. Modeling Approach](#3-modeling-approach)",
        "[3.1 Model Methodology](#31-model-methodology)",
        "[3.2 Model Interactions & Dependencies](#32-model-interactions--dependencies)",
        "[3.3 Model Assumptions](#33-model-assumptions)",
        "[3.4 Limitations](#34-limitations)",
        "[4. Model Estimation & Specification](#4-model-estimation--specification)",
        "[4.1 Estimation Approach, Algorithm Selection, and Configuration](#41-estimation-approach-algorithm-selection-and-configuration)",
        "[4.2 Model Segmentation](#42-model-segmentation)",
        "[4.3 Variable/Feature Selection Process](#43-variablefeature-selection-process)",
        "[4.4 Model Selection](#44-model-selection)",
        "[4.5 Final Model Specification](#45-final-model-specification)",
        "[4.5.1 Ensemble Functional Form](#451-ensemble-functional-form)",
        "[4.5.2 Recursive Forecast Logic](#452-recursive-forecast-logic)",
        "[4.5.3 Input Variables Required for Production](#453-input-variables-required-for-production)",
        "[4.5.4 Component Model Configuration](#454-component-model-configuration)",
        "[4.5.5 Feature Importance and Relative Influence](#455-feature-importance-and-relative-influence)",
        "[4.5.6 Error Terms and Prediction Intervals](#456-error-terms-and-prediction-intervals)",
        "[4.5.7 Production Implementation Requirements](#457-production-implementation-requirements)",
        "[4.5.8 Artifact and Provenance References](#458-artifact-and-provenance-references)",
        "[4.5.9 Step-by-Step Production Algorithm](#459-step-by-step-production-algorithm)",
        "[4.5.10 Forecast-Cycle Pseudocode](#4510-forecast-cycle-pseudocode)",
        "[4.5.11 Production Input / Output Contract](#4511-production-input--output-contract)",
        "[4.6 In-Model Overlays](#46-in-model-overlays)",
        "[5. Model Testing](#5-model-testing)",
        "[5.1 Model Development Test Plan & Approach](#51-model-development-test-plan--approach)",
        "[5.2 Backtesting](#52-backtesting)",
        "[5.3 Segmentation Testing](#53-segmentation-testing)",
        "[5.4 Sensitivity and Feature Importance Testing](#54-sensitivity-and-feature-importance-testing)",
        "[5.5 Scenario Analysis & Stress Testing](#55-scenario-analysis--stress-testing)",
        "[5.6 Benchmarking](#56-benchmarking)",
        "[5.7 Assumptions Analysis](#57-assumptions-analysis)",
        "[5.8 In-Model Overlay Testing](#58-in-model-overlay-testing)",
        "[6. Model Governance](#6-model-governance)",
        "[6.1 Model Risk Considerations](#61-model-risk-considerations)",
        "[6.2 Monitoring and Ongoing Performance Review](#62-monitoring-and-ongoing-performance-review)",
        "[6.3 Implementation and Control Considerations](#63-implementation-and-control-considerations)",
        "[7. Appendix](#7-appendix)",
        "[7.1 Glossary](#71-glossary)",
        "[7.2 Model Code & Dataset Location](#72-model-code-dataset-location)",
        "[7.3 References](#73-references)",
        "[7.4 Change Log](#74-change-log)",
        "[7.5 Data Dictionary](#75-data-dictionary)",
    ]
    return "## Table of Contents\n" + "  \n".join(entries)


def main() -> int:
    args = parse_args()
    ctx = load_context(Path(args.finalized_run_dir))
    content = "\n\n".join(
        [
            "# Model Development Document",
            table_of_contents(),
            section_overview(ctx),
            section_modeling_data(ctx),
            section_modeling_approach(ctx),
            section_estimation(ctx),
            section_testing(ctx),
            section_governance(ctx),
            section_appendix(ctx),
        ]
    )
    Path(args.output_path).write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
