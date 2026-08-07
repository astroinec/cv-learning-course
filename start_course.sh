#!/usr/bin/env bash
set -euo pipefail

COURSE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "$COURSE_DIR/.venv/bin/python" ]]; then
  PYTHON="$COURSE_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "未找到 Python。请先安装 Python 3.10+ 并按照 README 创建 .venv。" >&2
  exit 1
fi

cd "$COURSE_DIR"
exec "$PYTHON" -m jupyter lab .
