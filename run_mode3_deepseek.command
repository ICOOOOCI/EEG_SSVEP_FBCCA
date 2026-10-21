#!/bin/bash
# Mode 3: DeepSeek ChatBox
# First run: ./run_mode3_deepseek.command --configure-api-key
# Normal run: ./run_mode3_deepseek.command

set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR" || exit 10

export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export DEEPSEEK_API_URL="${DEEPSEEK_API_URL:-https://api.deepseek.com/chat/completions}"

ARGS=(
    --config "$SCRIPT_DIR/config/mode3_deepseek.toml"
    --fbcca-profile m3_5band
    --protocol standard_5s
    --response-delay 0.23
)

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    "$SCRIPT_DIR/.venv/bin/python" -m ssvep_fbcca "${ARGS[@]}" "$@"
elif command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m ssvep_fbcca "${ARGS[@]}" "$@"
else
    python3 -m ssvep_fbcca "${ARGS[@]}" "$@"
fi
EXIT_CODE=$?
printf 'Exit code: %s\n' "$EXIT_CODE"
exit "$EXIT_CODE"
