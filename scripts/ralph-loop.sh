#!/bin/bash
# ralph-loop.sh — Autonomous implement → evaluate → fix loop
#
# Usage: ./scripts/ralph-loop.sh [--max-cycles 50] [--workers 3] [--budget 5.00]
#
# Runs overnight unattended. Each cycle:
#   1. Spawns N worker agents in parallel (each picks one task from STM)
#   2. Waits for all workers to finish
#   3. Runs an evaluator agent to review work and create fix tasks
#   4. Checks convergence — stops when CONVERGED or max cycles reached
#
# State is persisted across invocations via:
#   - STM (task tracking)
#   - Git (code changes, one commit per task)
#   - ios/PageDewarp/.ralph-status (convergence flag from evaluator)
#
# Each `claude -p` invocation gets fresh context — no overflow.
# Requires: claude CLI, stm CLI, git
#
# Cost control:
#   --budget sets max USD per claude invocation (default $5.00)
#   --max-cycles caps total loop iterations (default 50)
#   Total max cost ≈ budget × (workers + 1) × max_cycles

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/scripts/ralph-logs"
STATUS_FILE="$PROJECT_DIR/.ralph-status"
WORKER_PROMPT="$SCRIPT_DIR/ralph-worker.md"
EVALUATOR_PROMPT="$SCRIPT_DIR/ralph-evaluator.md"

# Defaults
MAX_CYCLES=50
NUM_WORKERS=3
BUDGET_PER_RUN=5.00
CYCLE=0
MODEL="sonnet"  # Use sonnet for workers (fast + capable), opus for evaluator

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-cycles) MAX_CYCLES="$2"; shift 2 ;;
        --workers) NUM_WORKERS="$2"; shift 2 ;;
        --budget) BUDGET_PER_RUN="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_DIR/ralph.log"
}

run_worker() {
    local worker_id=$1
    local cycle=$2
    local log_file="$LOG_DIR/cycle-${cycle}-worker-${worker_id}.log"

    log "  Worker $worker_id starting..."

    # Each worker gets the prompt from file, runs non-interactively
    # --dangerously-skip-permissions: required for unattended operation
    # --max-budget-usd: cost cap per invocation
    # --model: use fast model for implementation work
    claude -p "$(cat "$WORKER_PROMPT")" \
        --dangerously-skip-permissions \
        --max-budget-usd "$BUDGET_PER_RUN" \
        --model "$MODEL" \
        > "$log_file" 2>&1 || true

    log "  Worker $worker_id finished"
}

run_evaluator() {
    local cycle=$1
    local log_file="$LOG_DIR/cycle-${cycle}-evaluator.log"

    log "  Evaluator starting..."

    # Evaluator uses opus for deeper reasoning
    claude -p "$(cat "$EVALUATOR_PROMPT")" \
        --dangerously-skip-permissions \
        --max-budget-usd "$BUDGET_PER_RUN" \
        --model opus \
        > "$log_file" 2>&1 || true

    log "  Evaluator finished"
}

check_convergence() {
    if [[ -f "$STATUS_FILE" ]]; then
        local status
        status=$(cat "$STATUS_FILE")
        if [[ "$status" == "CONVERGED" ]]; then
            return 0
        fi
    fi
    return 1
}

check_all_blocked() {
    # If all remaining tasks are blocked, we're stuck
    local pending
    pending=$(stm list 2>/dev/null | grep -c "pending" || echo "0")
    local blocked
    blocked=$(stm list 2>/dev/null | grep -c "blocked" || echo "0")
    local in_progress
    in_progress=$(stm list 2>/dev/null | grep -c "in_progress" || echo "0")

    if [[ "$pending" -eq 0 && "$in_progress" -eq 0 && "$blocked" -gt 0 ]]; then
        return 0  # all blocked
    fi
    return 1
}

# --- Pre-flight checks ---

command -v claude >/dev/null 2>&1 || { echo "Error: claude CLI not found"; exit 1; }
command -v stm >/dev/null 2>&1 || { echo "Error: stm CLI not found"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "Error: git not found"; exit 1; }

if [[ ! -f "$WORKER_PROMPT" ]]; then
    echo "Error: Worker prompt not found at $WORKER_PROMPT"
    exit 1
fi
if [[ ! -f "$EVALUATOR_PROMPT" ]]; then
    echo "Error: Evaluator prompt not found at $EVALUATOR_PROMPT"
    exit 1
fi

# --- Main loop ---

log "=== Ralph Loop starting ==="
log "Max cycles: $MAX_CYCLES, Workers per cycle: $NUM_WORKERS"
log "Budget per run: \$$BUDGET_PER_RUN, Worker model: $MODEL"
log "Project: $PROJECT_DIR"
log ""

cd "$PROJECT_DIR"

while [[ $CYCLE -lt $MAX_CYCLES ]]; do
    CYCLE=$((CYCLE + 1))
    log "--- Cycle $CYCLE / $MAX_CYCLES ---"

    # Check if we're stuck
    if check_all_blocked; then
        log "All remaining tasks are blocked. Stopping."
        log "Run 'stm list --pretty' to see blocked tasks."
        exit 1
    fi

    # Phase 1: Spawn workers in parallel
    log "Phase 1: Running $NUM_WORKERS worker(s)..."
    pids=()
    for i in $(seq 1 "$NUM_WORKERS"); do
        run_worker "$i" "$CYCLE" &
        pids+=($!)
        # Stagger worker starts by 2s to reduce STM race conditions
        sleep 2
    done

    # Wait for all workers
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    log "All workers done."

    # Phase 2: Run evaluator (sequential — needs to see all worker output)
    log "Phase 2: Running evaluator..."
    run_evaluator "$CYCLE"

    # Phase 3: Check convergence
    if check_convergence; then
        log ""
        log "=== CONVERGED after $CYCLE cycles ==="
        log "See logs in $LOG_DIR/"
        log ""
        log "Task summary:"
        stm list --pretty 2>/dev/null || stm list
        exit 0
    fi

    log "Not yet converged. Continuing..."
    log ""

    # Pause between cycles
    sleep 5
done

log ""
log "=== Reached max cycles ($MAX_CYCLES) without converging ==="
log "Check $LOG_DIR/ for details."
log ""
log "Task summary:"
stm list --pretty 2>/dev/null || stm list
exit 1
