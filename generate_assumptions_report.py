#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-spec-path", required=True)
    parser.add_argument("--feature-importance-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> int:
    args = parse_args()
    spec = read_json(Path(args.finalized_spec_path))
    importance = read_json(Path(args.feature_importance_path))
    payload = {
        "model_class": spec["model_class"],
        "method": spec.get("method"),
        "assumptions": [
            "Historical quarterly HPI and macro relationships are informative for future forecasting over the governed use horizon.",
            "The finalized ensemble uses fixed weights and does not adapt online after model approval.",
            "Forecast quality depends on the continued informativeness of lagged HPI and consumer_confidence, which dominate the selected component specifications.",
            "The validation framework based on annual rolling origins from 2005-12-31 through 2014-12-31 remains representative enough for model ranking.",
            "The national aggregate model is intended for national HPI forecasting and does not imply regional segmentation validity.",
        ],
        "limitations": [
            "The finalized champion currently has baseline forecast artifacts only; adverse and severely adverse scenario forecasts are not yet generated for the ensemble artifact.",
            "No in-model overlays are present in the finalized champion.",
            "Model relationships may degrade under structural breaks not well represented in the search sample.",
        ],
        "dominant_drivers": importance.get("components", []),
    }
    write_json(Path(args.output_path), payload)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
