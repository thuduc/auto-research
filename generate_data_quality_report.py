#!/usr/bin/env python3.12
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
SEARCH_PANEL_PATH = ROOT / "data" / "processed" / "search_panel.parquet"
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.json"
VARIABLES_PATH = ROOT / "config" / "variables.yaml"
DATA_QUALITY_PATH = ROOT / "data" / "processed" / "data_quality_report.json"
COLLINEARITY_PATH = ROOT / "data" / "processed" / "collinearity_report.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def vif_scores(frame: pd.DataFrame) -> dict[str, float]:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return {}

    clean = frame.dropna().copy()
    if clean.empty or clean.shape[1] < 2:
        return {}
    x = clean.to_numpy(dtype=float)
    scores: dict[str, float] = {}
    for index, column in enumerate(clean.columns):
        try:
            scores[column] = float(variance_inflation_factor(x, index))
        except Exception:
            scores[column] = float("nan")
    return scores


def main() -> int:
    search = pd.read_parquet(SEARCH_PANEL_PATH)
    manifest = load_json(MANIFEST_PATH)
    with VARIABLES_PATH.open("r", encoding="utf-8") as handle:
        variable_config = yaml.safe_load(handle)

    search["date"] = pd.to_datetime(search["date"])
    variable_keys = [item["key"] for item in manifest.get("variables", [])]
    included_columns = [column for column in [manifest.get("target_column", "hpi")] + variable_keys if column in search.columns]

    coverage: list[dict[str, Any]] = []
    for column in included_columns:
        series = search[column]
        non_null = int(series.notna().sum())
        first_valid = None
        last_valid = None
        if non_null:
            valid_dates = search.loc[series.notna(), "date"]
            first_valid = str(valid_dates.iloc[0].date())
            last_valid = str(valid_dates.iloc[-1].date())
        coverage.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": int(len(series)),
                "non_null_rows": non_null,
                "missing_rows": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "first_valid_date": first_valid,
                "last_valid_date": last_valid,
            }
        )

    numeric = search[included_columns].select_dtypes(include=[np.number]).copy()
    corr = numeric.corr(method="pearson") if not numeric.empty else pd.DataFrame()
    high_corr_pairs: list[dict[str, Any]] = []
    if not corr.empty:
        columns = list(corr.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and abs(float(value)) > 0.85:
                    high_corr_pairs.append({"left": left, "right": right, "correlation": float(value)})

    vif = vif_scores(numeric)

    quality_payload = {
        "search_start": manifest.get("search_start"),
        "search_end": manifest.get("search_end"),
        "primary_search_start": manifest.get("primary_search_start"),
        "target_column": manifest.get("target_column", "hpi"),
        "row_count": int(len(search)),
        "column_count": int(search.shape[1]),
        "included_variables": [item["key"] for item in manifest.get("variables", [])],
        "excluded_variables": manifest.get("excluded_variables", []),
        "coverage_summary": coverage,
        "dummy_variables": [item["key"] for item in variable_config.get("dummies", [])],
    }
    collinearity_payload = {
        "correlation_matrix": corr.round(6).to_dict() if not corr.empty else {},
        "vif_scores": vif,
        "high_correlation_pairs": high_corr_pairs,
    }

    write_json(DATA_QUALITY_PATH, quality_payload)
    write_json(COLLINEARITY_PATH, collinearity_payload)
    print(f"Wrote {DATA_QUALITY_PATH}")
    print(f"Wrote {COLLINEARITY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
