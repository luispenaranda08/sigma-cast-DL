"""Compute training-distribution statistics used for drift monitoring.

Loads `features.pkl` (produced by the feature-engineering notebook) and
writes `app/training_stats.json` with per-feature percentiles of the
training split for the (lag, fold) matching the deployed model.

The API uses these percentiles at inference time: an input value falling
outside [p1, p99] is flagged as out-of-distribution. Over many requests,
the fraction OOD flips /health to "degraded".

Usage
-----
    python -m scripts.compute_training_stats \\
        --features features.pkl \\
        --scalers app/scalers.joblib \\
        --out app/training_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("compute_training_stats")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("features.pkl"))
    parser.add_argument("--scalers", type=Path, default=Path("app/scalers.joblib"))
    parser.add_argument(
        "--results", type=Path, default=Path("results.pkl"),
        help="Used to resolve the (lag, fold) of the deployed model.",
    )
    parser.add_argument("--out", type=Path, default=Path("app/training_stats.json"))
    parser.add_argument("--lower-pct", type=float, default=1.0)
    parser.add_argument("--upper-pct", type=float, default=99.0)
    args = parser.parse_args(argv)

    if not args.features.exists():
        logger.error("%s not found.", args.features)
        return 2

    with open(args.features, "rb") as f:
        features = pickle.load(f)

    # Resolve which (lag, fold) the deployed model corresponds to.
    if args.scalers.exists():
        bundle = joblib.load(args.scalers)
        lag = int(bundle["lag"])
        model_type = str(bundle.get("model_type", "unknown"))
        fold = int(bundle.get("best_global", {}).get("fold", 0))
    elif args.results.exists():
        with open(args.results, "rb") as f:
            results = pickle.load(f)
        bg = results["best_global"]
        lag = int(bg["lag"])
        model_type = str(bg["mtype"])
        fold = int(bg["fold"])
    else:
        logger.error("Neither scalers.joblib nor results.pkl was found.")
        return 2

    splits = features["splits"]
    if lag not in splits:
        logger.error("lag=%d not present in features.pkl splits.", lag)
        return 3
    X_train = np.asarray(splits[lag]["X"][fold], dtype=np.float64)
    if X_train.ndim != 2:
        logger.error("Expected 2-D X_train; got shape %s", X_train.shape)
        return 3

    # Per-feature stats and a flattened overall range.
    flat = X_train.reshape(-1)
    p_lo = float(np.percentile(flat, args.lower_pct))
    p_hi = float(np.percentile(flat, args.upper_pct))

    payload = {
        "model_type": model_type,
        "lag": lag,
        "fold": fold,
        "feature_p1": round(p_lo, 8),
        "feature_p99": round(p_hi, 8),
        "feature_mean": round(float(flat.mean()), 8),
        "feature_std": round(float(flat.std()), 8),
        "feature_min": round(float(flat.min()), 8),
        "feature_max": round(float(flat.max()), 8),
        "n_training_samples": int(X_train.shape[0]),
        "percentile_range": [args.lower_pct, args.upper_pct],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote %s", args.out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
