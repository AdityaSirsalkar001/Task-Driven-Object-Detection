#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$PIPELINE_DIR/../../.." && pwd)"
cd "$PIPELINE_DIR"
exec "$PROJECT_ROOT/qwen_env/bin/python" -m unittest discover -s three_stage/tests -v
