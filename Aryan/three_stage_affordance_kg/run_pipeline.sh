#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$PIPELINE_DIR/../../.." && pwd)"
exec "$PROJECT_ROOT/qwen_env/bin/python" "$PIPELINE_DIR/run_pipeline.py" "$@"
