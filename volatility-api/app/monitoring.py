"""In-process telemetry and drift monitoring.

We keep a fixed-size ring buffer of recent predictions. `/health` consults
this buffer to compute the fraction of requests with OOD inputs; if that
fraction crosses the configured threshold the service reports
`status="degraded"` so that upstream monitors can alert.

This is intentionally in-memory only. For a multi-replica deployment this
would be replaced with Prometheus counters or a shared store; for a single-
replica container (the common student/demo setup), it's exactly right.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class PredictionEvent:
    timestamp: float
    lag_count: int
    any_ood: bool
    fraction_ood: float


class TelemetryStore:
    """Thread-safe ring buffer of recent predictions."""

    def __init__(self, buffer_size: int) -> None:
        self._buffer: Deque[PredictionEvent] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._total_predictions = 0

    # ── ingestion ────────────────────────────────────────────────────────
    def record(self, any_ood: bool, fraction_ood: float, lag_count: int) -> None:
        event = PredictionEvent(
            timestamp=time.time(),
            lag_count=lag_count,
            any_ood=bool(any_ood),
            fraction_ood=float(fraction_ood),
        )
        with self._lock:
            self._buffer.append(event)
            self._total_predictions += 1

    # ── snapshots ────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            n = len(self._buffer)
            ood_pct = (
                100.0 * sum(1 for e in self._buffer if e.any_ood) / n
                if n > 0
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self._started_at, 3),
                "predictions_total": self._total_predictions,
                "predictions_last_window": n,
                "drift_pct_out_of_distribution": round(ood_pct, 3),
            }

    def reset(self) -> None:
        """Used by tests to isolate state across cases."""
        with self._lock:
            self._buffer.clear()
            self._total_predictions = 0
            self._started_at = time.time()


def evaluate_status(
    drift_pct: float, threshold_pct: float, model_loaded: bool
) -> tuple[str, list[str]]:
    """Decide overall health status from telemetry.

    Returns (status, degraded_reasons).
    """
    if not model_loaded:
        return "unhealthy", ["model_not_loaded"]
    reasons: list[str] = []
    if drift_pct > threshold_pct:
        reasons.append(
            f"input_drift_exceeds_threshold ({drift_pct:.1f}% > "
            f"{threshold_pct:.1f}%)"
        )
    status = "degraded" if reasons else "healthy"
    return status, reasons
