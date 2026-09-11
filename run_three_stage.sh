#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

exec "$PROJECT_ROOT/qwen_env/bin/python" \
  "$REPO_ROOT/Aryan/three_stage_affordance_kg/run_pipeline.py" "$@"
