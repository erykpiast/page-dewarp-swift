#!/bin/bash
# ralph-status.sh — Check the status of a running Ralph loop
#
# Usage: ./scripts/ralph-status.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/scripts/ralph-logs"

echo "=== Ralph Loop Status ==="
echo ""

# Current cycle from log
if [[ -f "$LOG_DIR/ralph.log" ]]; then
    echo "Last log entries:"
    tail -10 "$LOG_DIR/ralph.log"
    echo ""
fi

# Convergence status
STATUS_FILE="$PROJECT_DIR/.ralph-status"
if [[ -f "$STATUS_FILE" ]]; then
    echo "Convergence: $(cat "$STATUS_FILE")"
else
    echo "Convergence: not yet checked"
fi
echo ""

# STM summary
echo "Task summary:"
cd "$PROJECT_DIR"
stm list --pretty 2>/dev/null || stm list 2>/dev/null || echo "STM not available"
echo ""

# Git summary
echo "Recent commits:"
git -C "$PROJECT_DIR" log --oneline -10 2>/dev/null || echo "No commits yet"
echo ""

# Log file count
if [[ -d "$LOG_DIR" ]]; then
    echo "Log files: $(find "$LOG_DIR" -name '*.log' -newer "$LOG_DIR/ralph.log" 2>/dev/null | wc -l | tr -d ' ') (current cycle)"
    echo "Total log files: $(find "$LOG_DIR" -name '*.log' 2>/dev/null | wc -l | tr -d ' ')"
fi
