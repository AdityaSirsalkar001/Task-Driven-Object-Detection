#!/usr/bin/env python3
"""Single entry point for Aryan's three-stage research pipeline."""
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
BASELINE_HELPERS = PIPELINE_ROOT.parent / "qwen_then_florence_baseline"
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(BASELINE_HELPERS))

from three_stage.__main__ import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
