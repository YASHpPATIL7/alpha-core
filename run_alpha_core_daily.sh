#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_alpha_core_daily.sh
# Alpha-Core daily pipeline runner
#
# What this runs:
#   Step 0: DCC freshness check — if Vajra DCC pkl > 5 days old, auto-refresh
#   Step 1: Full Alpha-Core pipeline (M1-M10) → signals → Alpaca gate
#   Step 2: Alpaca execution gate (isolated submission log)
#
#   Runs at 4:00 PM IST = 10:30 AM ET = 30 min before US market close
#   (ETF proxy orders fill on same-day market close)
#
# Install cron (run once):
#   crontab -e
#   Add line:  30 10 * * 1-5 /Users/yashpatil/Local_Mark1/alpha-core/run_alpha_core_daily.sh
#   (Mon-Fri only — US market days)
#
# Manual run:
#   bash /Users/yashpatil/Local_Mark1/alpha-core/run_alpha_core_daily.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="/Users/yashpatil/Local_Mark1/alpha-core"
VAJRA_DIR="/Users/yashpatil/Local_Mark1/indian-risk-engine"
VENV="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
TODAY=$(date +%Y-%m-%d)
LOGFILE="$LOG_DIR/alpha_core_${TODAY}.log"

# Ensure alpha_core package is importable for standalone module calls
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR"

# ── Header ────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════" >> "$LOGFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S IST')] ALPHA-CORE DAILY PIPELINE" >> "$LOGFILE"
echo "══════════════════════════════════════════════════" >> "$LOGFILE"

# ── Step 0: DCC freshness check ───────────────────────────────
# If Vajra's DCC pkl is older than 5 days, auto-refresh.
# This keeps the regime-aware covariance current without manual intervention.
DCC_PKL="$VAJRA_DIR/data/vajra_dcc_cov.pkl"
DCC_MAX_AGE_DAYS=5

echo "[$(date '+%H:%M:%S')] Checking DCC pkl freshness (max ${DCC_MAX_AGE_DAYS} days)" >> "$LOGFILE"

if [ -f "$DCC_PKL" ]; then
    # mtime in seconds since epoch
    PKL_MTIME=$(python3 -c "import os; print(int(os.path.getmtime('$DCC_PKL')))")
    NOW=$(python3 -c "import time; print(int(time.time()))")
    AGE_DAYS=$(( (NOW - PKL_MTIME) / 86400 ))
    echo "[$(date '+%H:%M:%S')] DCC pkl age: ${AGE_DAYS} day(s)" >> "$LOGFILE"
else
    AGE_DAYS=999
    echo "[$(date '+%H:%M:%S')] DCC pkl not found — will rebuild" >> "$LOGFILE"
fi

if [ "$AGE_DAYS" -ge "$DCC_MAX_AGE_DAYS" ]; then
    echo "[$(date '+%H:%M:%S')] DCC stale — refreshing Vajra GARCH + DCC..." >> "$LOGFILE"

    # Use Vajra's own venv if it exists, otherwise fall back to the alpha-core venv
    VAJRA_VENV="$VAJRA_DIR/venv/bin/python"
    if [ ! -f "$VAJRA_VENV" ]; then
        VAJRA_VENV="$VENV"
    fi

    # Run GARCH estimation (writes vajra_sigma.pkl, vajra_z.pkl, vajra_returns.csv)
    if "$VAJRA_VENV" "$VAJRA_DIR/risk_engine/garch_model.py" >> "$LOGFILE" 2>&1; then
        echo "[$(date '+%H:%M:%S')] GARCH refresh OK" >> "$LOGFILE"
    else
        echo "[$(date '+%H:%M:%S')] GARCH refresh FAILED — continuing with existing pkl" >> "$LOGFILE"
    fi

    # Run DCC estimation (reads GARCH outputs, writes vajra_dcc_cov.pkl)
    if "$VAJRA_VENV" "$VAJRA_DIR/risk_engine/dcc_engine.py" >> "$LOGFILE" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DCC refresh OK — $(date '+%Y-%m-%d') pkl written" >> "$LOGFILE"
    else
        echo "[$(date '+%H:%M:%S')] DCC refresh FAILED — continuing with stale pkl" >> "$LOGFILE"
    fi
else
    echo "[$(date '+%H:%M:%S')] DCC fresh (${AGE_DAYS}d < ${DCC_MAX_AGE_DAYS}d) — skipping refresh" >> "$LOGFILE"
fi


echo "[$(date '+%H:%M:%S')] START main.py (M1-M10)" >> "$LOGFILE"

"$VENV" "$PROJECT_DIR/main.py" --skip-finbert >> "$LOGFILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] main.py FAILED (exit $EXIT_CODE)" >> "$LOGFILE"
    echo "[$(date '+%H:%M:%S')] ABORTING — not submitting orders with stale signals" >> "$LOGFILE"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] main.py OK" >> "$LOGFILE"

# ── Step 2: Alpaca execution gate (M10 standalone refresh) ────
# main.py already runs M10, but this gives a clean isolated log entry
echo "[$(date '+%H:%M:%S')] START alpaca_gate.py (order submission)" >> "$LOGFILE"

"$VENV" -m alpha_core.alpaca_gate >> "$LOGFILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] alpaca_gate.py FAILED (exit $EXIT_CODE)" >> "$LOGFILE"
    # Don't exit 1 here — signals were computed, just submission failed
    # Orders can be placed manually if needed
fi

echo "[$(date '+%H:%M:%S')] alpaca_gate.py done" >> "$LOGFILE"

# ── Done ──────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Pipeline complete" >> "$LOGFILE"
echo "══════════════════════════════════════════════════" >> "$LOGFILE"

# Keep only last 30 log files (1 month of daily runs)
ls -t "$LOG_DIR"/alpha_core_*.log 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
