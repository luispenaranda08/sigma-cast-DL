"""Baseline-vs-model statistical comparison.

`baselines.json` is produced offline by `scripts/compute_baselines.py`
using the test-set predictions stored in `results.pkl`. This module
loads the artifact and exposes it to the API.

The comparison uses the Diebold-Mariano test on squared-error loss
differentials. Null hypothesis: the two forecasts have equal expected
loss. A p-value < 0.05 lets us reject equality — so the model and naive
baseline differ significantly.

We also report RMSE improvement so downstream consumers have an effect-
size number alongside the p-value (an improvement can be statistically
significant but practically tiny, or vice versa).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from app.config import settings
from app.schemas import BaselineComparison

logger = logging.getLogger(__name__)


def load_baseline_comparison(
    path: Path | None = None,
) -> Optional[BaselineComparison]:
    """Return the baseline comparison, or None if the artifact is missing."""
    path = path or settings.baselines_path
    if not path.exists():
        logger.warning(
            "baselines.json not found at %s — run "
            "scripts/compute_baselines.py to generate it.",
            path,
        )
        return None

    try:
        with open(path) as f:
            payload = json.load(f)
        return BaselineComparison(**payload)
    except Exception as e:  # pragma: no cover — defensive
        logger.error("Failed to load baseline comparison: %s", e)
        return None
