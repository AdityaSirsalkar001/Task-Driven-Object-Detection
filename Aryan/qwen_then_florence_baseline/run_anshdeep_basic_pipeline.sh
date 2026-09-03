#!/usr/bin/env bash
# Run the adapted Anshdeep Qwen -> Florence two-stage baseline.
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PROJECT_DIR/qwen_env/bin/python" "$PROJECT_DIR/anshdeep_basic_pipeline_runner.py" "$@"
