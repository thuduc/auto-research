#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
from pathlib import Path

import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-path", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-class", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = Path(args.spec_path)
    with spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    train.set_active_spec(payload["spec"])
    trial_args = argparse.Namespace(
        mode="finalize",
        model_class=args.model_class,
        experiment_id=args.experiment_id,
        trial_id=args.trial_id,
        output_dir=args.output_dir,
        max_trials=300,
        improvement_threshold=0.005,
        patience=40,
    )
    return train.run_single_trial(trial_args)


if __name__ == "__main__":
    raise SystemExit(main())
