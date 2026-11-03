#!/bin/bash
# 离线：./start_keyboard_dual_mode.command --offline-session 会话.zip --offline-output 输出目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR" || exit 10
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_ARGS=(
    --config "$SCRIPT_DIR/config/default.toml"
    --response-delay 0.23
    --fullscreen
)
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    FBCCA_PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then
    FBCCA_PYTHON="$(command -v python3.10)"
else
    FBCCA_PYTHON="$(command -v python3)"
fi
run_keyboard() {
    # 可选的依赖检查脚本；项目里没有时直接启动（mne-lsl 缺失时程序会给出明确报错）。
    if [[ -f "$SCRIPT_DIR/ensure_mne_lsl.py" ]]; then
        "$FBCCA_PYTHON" "$SCRIPT_DIR/ensure_mne_lsl.py" "$@" || return $?
    fi
    "$FBCCA_PYTHON" -m ssvep_fbcca "${DEFAULT_ARGS[@]}" "$@"
}
run_keyboard "$@"
EXIT_CODE=$?
printf 'Exit code: %s\n' "$EXIT_CODE"
if [[ -t 0 && "${FBCCA_NO_PAUSE:-0}" != 1 ]]; then
    read -r -p 'Press Enter to close...'
fi
exit "$EXIT_CODE"
