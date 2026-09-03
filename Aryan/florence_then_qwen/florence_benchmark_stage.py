"""One-image Florence stage runner, executed only with edge_project/florence_env."""

import argparse
import json
from pathlib import Path

from edge_project import florence_single_image as florence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Florence object detection for one benchmark image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--boxed-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw, parsed = florence.run_task(args.image_path, "<OD>")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.boxed_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps({"raw": raw, "parsed": parsed}, indent=2))
    florence.render_od_visualization(args.image_path, parsed, args.boxed_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
