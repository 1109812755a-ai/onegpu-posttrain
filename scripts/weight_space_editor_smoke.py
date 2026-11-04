"""Call Bailian once only when explicitly invoked with --live."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onegpu_posttrain.weight_space.smoke import live_editor_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in Qwen3.8-Max editor connectivity check")
    parser.add_argument("--live", action="store_true", help="make one billable provider request")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required before sending a provider request")
    report = live_editor_smoke(args.workspace)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
