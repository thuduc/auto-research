#!/usr/bin/env python3.12
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from fredapi import Fred
except ImportError:  # pragma: no cover
    Fred = None


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "variables.yaml"
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
SEARCH_PANEL_PATH = PROCESSED_DATA_DIR / "search_panel.parquet"
HOLDOUT_PANEL_PATH = PROCESSED_DATA_DIR / "holdout_panel.parquet"
MANIFEST_PATH = PROCESSED_DATA_DIR / "dataset_manifest.json"
BACKTEST_FIRST_ORIGIN = pd.Timestamp("2005-12-31")
HOLDOUT_QUARTERS = 4
FAR_HORIZON_QUARTERS = 40


@dataclass(frozen=True)
class VariableSpec:
    key: str
    transform: str
    series_id: str | None = None
    fred_url: str | None = None
    group: str | None = None
    expected_sign: str | None = None
    dummy_type: str | None = None
    start: str | None = None
    date: str | None = None


def ensure_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_variable_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_specs(config: dict[str, Any]) -> tuple[dict[str, Any], list[VariableSpec], list[VariableSpec]]:
    target = config["target"]
    exogenous = [
        VariableSpec(
            key=item["key"],
            series_id=item["series_id"],
            fred_url=item.get("fred_url"),
            group=item.get("group"),
            transform=item["transform"],
            expected_sign=item.get("expected_sign"),
        )
        for item in config["exogenous"]
    ]
    dummies = [
        VariableSpec(
            key=item["key"],
            transform="level",
            dummy_type=item["type"],
            start=item.get("start"),
            date=item.get("date"),
        )
        for item in config["dummies"]
    ]
    return target, exogenous, dummies


def make_fred_client() -> Fred:
    if Fred is None:
        raise RuntimeError("fredapi is not installed. Install dependencies from requirements/econometric.txt.")
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set.")
    return Fred(api_key=api_key)


def to_quarter_end_index(series: pd.Series) -> pd.Series:
    data = series.dropna().astype(float).sort_index()
    index = pd.to_datetime(data.index)
    data.index = index
    quarterly = data.resample("QE-DEC").mean()
    quarterly.index = quarterly.index.to_period("Q").to_timestamp("Q")
    return quarterly


def fetch_fred_series(fred: Fred, spec: VariableSpec) -> pd.Series:
    raw = fred.get_series(spec.series_id)
    series = pd.Series(raw)
    quarterly = to_quarter_end_index(series)
    quarterly.to_csv(RAW_DATA_DIR / f"{spec.key}.csv", header=["value"])
    quarterly.name = spec.key
    return quarterly


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return series
    if transform == "log_level":
        return np.log(series.where(series > 0))
    if transform == "log_diff":
        return np.log(series.where(series > 0)).diff()
    raise ValueError(f"Unsupported transform: {transform}")


def generate_dummy(index: pd.DatetimeIndex, spec: VariableSpec) -> pd.Series:
    if spec.dummy_type == "step":
        start = pd.Timestamp(spec.start)
        values = (index >= start).astype(int)
    elif spec.dummy_type == "pulse":
        date = pd.Timestamp(spec.date)
        values = (index == date).astype(int)
    else:
        raise ValueError(f"Unsupported dummy type: {spec.dummy_type}")
    return pd.Series(values, index=index, name=spec.key, dtype=float)


def longest_missing_run(series: pd.Series) -> int:
    longest = current = 0
    for is_missing in series.isna().tolist():
        if is_missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_panel(
    target_cfg: dict[str, Any],
    exogenous: list[VariableSpec],
    dummies: list[VariableSpec],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fred = make_fred_client()
    primary_search_start = pd.Timestamp(target_cfg["primary_search_start"])
    target_series = fetch_fred_series(
        fred,
        VariableSpec(
            key=target_cfg["key"],
            series_id=target_cfg["series_id"],
            fred_url=target_cfg.get("fred_url"),
            transform="level",
        ),
    )

    columns: list[pd.Series] = [target_series.rename("hpi")]
    excluded_variables: list[dict[str, Any]] = []
    for spec in exogenous:
        raw_series = fetch_fred_series(fred, spec)
        transformed = apply_transform(raw_series, spec.transform).rename(spec.key)
        first_valid = transformed.dropna().index.min()
        if first_valid is None or first_valid > primary_search_start:
            excluded_variables.append(
                {
                    "key": spec.key,
                    "series_id": spec.series_id,
                    "reason": "insufficient_history_for_primary_search_window",
                    "first_valid_date": None if first_valid is None else str(first_valid.date()),
                }
            )
            continue
        columns.append(transformed)

    panel = pd.concat(columns, axis=1, sort=True).sort_index()
    for spec in dummies:
        panel[spec.key] = generate_dummy(panel.index, spec)

    first_valid = max(panel[column].first_valid_index() for column in panel.columns if panel[column].first_valid_index() is not None)
    panel = panel.loc[first_valid:].copy()
    panel = panel.interpolate(method="linear", limit=2, limit_area="inside")

    long_gaps = {column: longest_missing_run(panel[column]) for column in panel.columns}
    bad_gaps = {column: gap for column, gap in long_gaps.items() if gap > 2}
    if bad_gaps:
        raise RuntimeError(f"Found missing runs longer than 2 quarters: {bad_gaps}")

    panel = panel.dropna().copy()
    if panel.empty:
        raise RuntimeError("Panel is empty after dropping residual missing rows.")

    panel["hpi_logdiff"] = np.log(panel["hpi"].where(panel["hpi"] > 0)).diff()
    panel = panel.dropna().copy()
    panel.index.name = "date"
    return panel.reset_index(), excluded_variables


def build_splits(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(panel) <= HOLDOUT_QUARTERS:
        raise RuntimeError("Panel is too short to create a 4-quarter holdout.")
    search_panel = panel.iloc[:-HOLDOUT_QUARTERS].reset_index(drop=True)
    holdout_panel = panel.iloc[-HOLDOUT_QUARTERS:].reset_index(drop=True)
    return search_panel, holdout_panel


def build_backtest_origins(search_panel: pd.DataFrame, primary_search_start: str) -> list[str]:
    search_with_index = search_panel.copy()
    search_with_index["date"] = pd.to_datetime(search_with_index["date"])
    eligible = search_with_index.loc[search_with_index["date"] >= pd.Timestamp(primary_search_start), "date"]
    if eligible.empty:
        raise RuntimeError("No search-sample observations remain after the primary search start.")
    final_origin = eligible.max() - pd.offsets.QuarterEnd(FAR_HORIZON_QUARTERS)
    if final_origin < BACKTEST_FIRST_ORIGIN:
        return []
    origin_dates = eligible[(eligible >= BACKTEST_FIRST_ORIGIN) & (eligible <= final_origin)]
    sampled = origin_dates.iloc[::4]
    return [timestamp.strftime("%Y-%m-%d") for timestamp in sampled]


def write_manifest(
    target_cfg: dict[str, Any],
    exogenous: list[VariableSpec],
    dummies: list[VariableSpec],
    search_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame,
    excluded_variables: list[dict[str, Any]],
) -> None:
    manifest = {
        "primary_search_start": target_cfg["primary_search_start"],
        "search_start": str(pd.to_datetime(search_panel["date"].min()).date()),
        "search_end": str(pd.to_datetime(search_panel["date"].max()).date()),
        "holdout_start": str(pd.to_datetime(holdout_panel["date"].min()).date()),
        "holdout_end": str(pd.to_datetime(holdout_panel["date"].max()).date()),
        "backtest_first_origin": str(BACKTEST_FIRST_ORIGIN.date()),
        "backtest_origins": build_backtest_origins(search_panel, target_cfg["primary_search_start"]),
        "near_horizons": [1, 12],
        "far_horizons": [13, 40],
        "target_column": "hpi",
        "target_growth_column": "hpi_logdiff",
        "variables": [
            {
                "key": spec.key,
                "series_id": spec.series_id,
                "fred_url": spec.fred_url,
                "transform": spec.transform,
                "group": spec.group,
                "expected_sign": spec.expected_sign,
            }
            for spec in exogenous
            if spec.key not in {item["key"] for item in excluded_variables}
        ],
        "excluded_variables": excluded_variables,
        "dummies": [
            {
                "key": spec.key,
                "type": spec.dummy_type,
                "start": spec.start,
                "date": spec.date,
            }
            for spec in dummies
        ],
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def main() -> None:
    ensure_dirs()
    config = load_variable_config()
    target_cfg, exogenous, dummies = build_specs(config)
    panel, excluded_variables = build_panel(target_cfg, exogenous, dummies)
    search_panel, holdout_panel = build_splits(panel)
    search_panel.to_parquet(SEARCH_PANEL_PATH, index=False)
    holdout_panel.to_parquet(HOLDOUT_PANEL_PATH, index=False)
    write_manifest(target_cfg, exogenous, dummies, search_panel, holdout_panel, excluded_variables)
    print(f"Wrote {SEARCH_PANEL_PATH}")
    print(f"Wrote {HOLDOUT_PANEL_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    if excluded_variables:
        print(f"Excluded variables: {[item['key'] for item in excluded_variables]}")


if __name__ == "__main__":
    main()
