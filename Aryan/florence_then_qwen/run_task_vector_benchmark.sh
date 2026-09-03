#!/usr/bin/env bash
# Run the two-stage Florence -> Qwen2.5-VL benchmark without activating qwen_env.
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PROJECT_DIR/qwen_env/bin/python" "$PROJECT_DIR/task_vector_benchmark.py" "$@"
