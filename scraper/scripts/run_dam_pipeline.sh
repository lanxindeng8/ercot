#!/bin/bash
# DAM Pipeline — daily orchestrator running at 14:00 CT
# Runs sequentially: predictions → API fetch → CDR fetch → Telegram notification
# Each step depends on the previous, so they run in order.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ERCOT_ROOT="$(dirname "$PROJECT_DIR")"
cd "$ERCOT_ROOT"

PYTHON_PRED="${HOME}/.venvs/ercot-prediction/bin/python"
PYTHON_SCRAPER="${HOME}/.venvs/ercot-scraper/bin/python"

mkdir -p "$HOME/logs/trueflux"
LOG="$HOME/logs/trueflux/dam-pipeline.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG"; }

# Step 1: Generate DAM predictions
log "Step 1: Generating DAM predictions..."
"$PYTHON_PRED" "$ERCOT_ROOT/prediction/scripts/generate_dam_predictions.py" 2>&1 | tee -a "$LOG"
STEP1=$?
log "Step 1 done (exit=$STEP1)"

# Step 2: Fetch DAM LMP from ERCOT API
log "Step 2: Fetching DAM LMP from API..."
cd "$PROJECT_DIR/src" && "$PYTHON_SCRAPER" scraper_dam_lmp.py 2>&1 | tee -a "$LOG"
STEP2=$?
log "Step 2 done (exit=$STEP2)"
cd "$ERCOT_ROOT"

# Step 3: Fetch DAM LMP from CDR HTML
log "Step 3: Fetching DAM LMP from CDR..."
cd "$PROJECT_DIR/src" && "$PYTHON_SCRAPER" scraper_dam_lmp_cdr.py 2>&1 | tee -a "$LOG"
STEP3=$?
log "Step 3 done (exit=$STEP3)"
cd "$ERCOT_ROOT"

# Step 4: Send Telegram DAM schedule
log "Step 4: Sending Telegram DAM schedule..."
"$PYTHON_PRED" "$ERCOT_ROOT/prediction/scripts/telegram_dam_schedule.py" 2>&1 | tee -a "$LOG"
STEP4=$?
log "Step 4 done (exit=$STEP4)"

log "DAM pipeline complete (steps: $STEP1/$STEP2/$STEP3/$STEP4)"
