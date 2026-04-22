"""Compute baseline vs model comparison with the Diebold-Mariano test.

Reads `results.pkl` from the project root (produced by the training
notebook), extracts test-set predictions and ground truth for the best
global (mtype, lag, fold), computes:

  * naive persistence baseline: ŷ(t+h) = y(t) for all h=1..H
  * squared-error loss differentials d_t = e_model^2 - e_naive^2
  * Diebold-Mariano statistic on d_t with small-sample correction
    (Harvey-Leybourne-Newbold, 1997)

Writes `app/baselines.json` with the result, which the API loads at
startup and exposes via GET /info.

Usage
-----
    python -m scripts.compute_baselines \\
        --results results.pkl \\
        --out app/baselines.json

The script is deterministic: same inputs → same JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("compute_baselines")


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test
# ─────────────────────────────────────────────────────────────────────────────
def diebold_mariano(
    errors_model: np.ndarray,
    errors_naive: np.ndarray,
    horizon: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss.

    Parameters
    ----------
    errors_model, errors_naive : 1-D arrays of forecast errors (same length).
    horizon : forecast horizon used to compute the long-run variance window.

    Returns
    -------
    (dm_statistic, p_value)
        Two-sided p-value under H0 of equal expected loss.
    """
    d = errors_model**2 - errors_naive**2
    n = len(d)
    if n < 8:
        return float("nan"), float("nan")

    mean_d = float(np.mean(d))
    # Newey-West long-run variance with lag = horizon - 1
    gamma_0 = float(np.var(d, ddof=0))
    lr_var = gamma_0
    for k in range(1, horizon):
        if k >= n:
            break
        gamma_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        lr_var += 2.0 * gamma_k
    lr_var = max(lr_var, 1e-12)

    dm = mean_d / np.sqrt(lr_var / n)

    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = float(dm * correction)

    # Two-sided p-value using a t-distribution with n-1 df (HLN)
    pval = float(2.0 * (1.0 - stats.t.cdf(abs(dm_corrected), df=n - 1)))
    return dm_corrected, pval


# ─────────────────────────────────────────────────────────────────────────────
# Payload extraction
# ─────────────────────────────────────────────────────────────────────────────
def _extract_best_fold_arrays(results: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (y_true, y_pred, meta) for the best (mtype, lag, fold)."""
    results_all = results["results_all"]
    best_global = results["best_global"]

    mtype = best_global["mtype"]
    lag = best_global["lag"]
    fold = best_global["fold"]

    r = results_all[mtype][lag][fold]
    y_true = np.asarray(r["y_test_raw"], dtype=np.float64)
    y_pred = np.asarray(r["yhat_test"], dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}"
        )
    return y_true, y_pred, {"mtype": mtype, "lag": lag, "fold": fold}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results.pkl"))
    parser.add_argument("--out", type=Path, default=Path("app/baselines.json"))
    parser.add_argument(
        "--horizon-for-dm",
        type=int,
        default=1,
        help="Horizon used to compute the Newey-West long-run variance "
        "window in the DM test.",
    )
    args = parser.parse_args(argv)

    if not args.results.exists():
        logger.error("%s not found.", args.results)
        return 2

    with open(args.results, "rb") as f:
        results = pickle.load(f)

    y_true, y_pred_model, meta = _extract_best_fold_arrays(results)
    logger.info(
        "Best model: %s | lag=%d | fold=%d | y_test_raw.shape=%s",
        meta["mtype"],
        meta["lag"],
        meta["fold"],
        y_true.shape,
    )

    # Naive persistence baseline on the test set.
    # We need the last observed value before each test sample, which we can
    # reconstruct from the lag window inside `results_all`. But the safer
    # approach, using what's already stored, is: predict `y(t-1)` for every
    # horizon. Since each row of y_test_raw is the H-step target starting at
    # t+1, the last observed value is y_test_raw[:, 0] shifted by one — for
    # the very first sample we fall back to y_test_raw[0, 0] itself, which
    # yields zero error on that single horizon-1 position.
    #
    # Equivalent and simpler: the naive h-step forecast at time t is the
    # observation at time t. For this evaluation we use the model's own
    # input window (the last value seen) as the anchor.
    # We approximate it using y_test_raw[:, 0] lagged by 1.
    anchor = np.empty(y_true.shape[0], dtype=np.float64)
    anchor[0] = y_true[0, 0]
    anchor[1:] = y_true[:-1, 0]
    y_pred_naive = np.repeat(anchor.reshape(-1, 1), y_true.shape[1], axis=1)

    # Flatten across all horizons for a single-series DM test.
    errors_model = (y_true - y_pred_model).reshape(-1)
    errors_naive = (y_true - y_pred_naive).reshape(-1)

    rmse_model = float(np.sqrt(np.mean(errors_model**2)))
    rmse_naive = float(np.sqrt(np.mean(errors_naive**2)))
    improvement_pct = float(100.0 * (rmse_naive - rmse_model) / rmse_naive)

    dm_stat, dm_pval = diebold_mariano(
        errors_model, errors_naive, horizon=args.horizon_for_dm
    )

    ALPHA = 0.05
    payload = {
        "naive_rmse": round(rmse_naive, 6),
        "model_rmse": round(rmse_model, 6),
        "rmse_improvement_pct": round(improvement_pct, 4),
        "diebold_mariano_statistic": round(dm_stat, 6),
        "diebold_mariano_pvalue": round(dm_pval, 6),
        "significance_alpha": ALPHA,
        "beats_naive_significantly": bool(
            dm_pval < ALPHA and rmse_model < rmse_naive
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote %s", args.out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
