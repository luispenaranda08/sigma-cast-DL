# ─────────────────────────────────────────────────────────────────────────
#   Volatility Forecasting API — production image
# ─────────────────────────────────────────────────────────────────────────
#
# Design choices:
#   * python:3.11-slim keeps the final image ~200 MB smaller than the full tag
#     while still giving us glibc for TensorFlow's CPU wheels.
#   * A dedicated non-root user (`appuser`) runs the process. This is the
#     single most common hardening miss in container images; costs nothing.
#   * HEALTHCHECK uses the same /health endpoint the orchestrator would use,
#     so `docker ps` and Kubernetes probes agree on what "ready" means.
#   * Dependencies are installed before the code is copied, so that editing
#     app/ or tests/ doesn't invalidate the pip layer cache.
#
FROM python:3.11-slim AS runtime

# ── Image metadata (OCI) ──────────────────────────────────────────────────
LABEL org.opencontainers.image.title="volatility-api" \
      org.opencontainers.image.description="7-day Bitcoin realized-volatility forecasting service" \
      org.opencontainers.image.source="https://github.com/<your-org>/volatility-api" \
      org.opencontainers.image.licenses="MIT"

# ── Python runtime tuning ─────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# ── Non-root user ─────────────────────────────────────────────────────────
RUN groupadd --system --gid 1001 appgroup \
 && useradd --system --uid 1001 --gid appgroup --home /app --shell /bin/bash appuser

WORKDIR /app

# ── Deps first (cached independently of source changes) ──────────────────
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

# ── Application code + artifacts ──────────────────────────────────────────
COPY app/        ./app/
COPY scripts/    ./scripts/
COPY pyproject.toml ./

# Pre-compile bytecode for faster startup
RUN python -m compileall -q app scripts

# Ownership and permissions
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; \
r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); \
sys.exit(0 if r.status == 200 else 1)" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
