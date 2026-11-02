"""CLI registration for local policy workflows; no optional UI imports."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .data.policy import load_policy
from .data.policy_build import build_policy_artifacts, execute_policy, load_source
from .data.policy_experiment import (
    compare_experiment,
    load_experiment,
    register_experiment,
)
from .reporting.data_report import write_report


def _run_policy(args) -> int:
    try:
        policy = load_policy(args.policy)
        if args.command == "validate-data-policy":
            result = policy.artifact()
        else:
            source = load_source(args.source)
            if args.command == "build-policy-data":
                result = build_policy_artifacts(policy, source, output_dir=args.output)
                result = {key: value for key, value in result.items() if key != "report"} | {
                    "selected": result["report"]["selected"], "stop_reason": result["report"]["stop_reason"]}
            else:
                _, result = execute_policy(policy, source, preview_rows=args.rows)
                if args.output:
                    if Path(args.output).exists():
                        raise FileExistsError("不覆盖已有预览报告，请使用新路径")
                    write_report(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "invalid_or_unavailable", "error": str(exc)}, ensure_ascii=False))
        return 2


def register_policy_commands(sub) -> None:
    for name in ("register-data-experiment", "compare-data-experiment"):
        parser = sub.add_parser(name, help="Register or compare a reproducible policy experiment")
        parser.add_argument("--config" if name.startswith("register") else "--experiment", required=True)
        parser.add_argument("--root", default=".", help="Explicit artifact/evidence root")
        parser.add_argument("--output", required=True)
        if name.startswith("compare"):
            parser.add_argument("--selections", help="JSON: {a:[{run_id,eval_id}], b:[...]} for existing evidence")
        parser.set_defaults(func=_run_experiment)
    for name, help_text in (
        ("validate-data-policy", "Validate a versioned SFT/Preference policy"),
        ("preview-data-policy", "Preview rules on a bounded local source; never download"),
        ("build-policy-data", "Build new policy datasets from verified local inputs"),
    ):
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("--policy", required=True)
        if name != "validate-data-policy":
            parser.add_argument("--source", required=True, help="Trusted local source YAML; paths relative to this file")
            parser.add_argument("--output", required=name == "build-policy-data")
        if name == "preview-data-policy":
            parser.add_argument("--rows", type=int, default=200, help="1..1000 raw rows; deterministic prefix, not a full-source estimate")
        parser.set_defaults(func=_run_policy)


def _run_experiment(args) -> int:
    try:
        if args.command == "register-data-experiment":
            plan = load_experiment(args.config)
            result = register_experiment(plan, root=args.root, output=args.output)
        else:
            plan = load_experiment(args.experiment)
            registration = yaml.safe_load(Path(args.experiment).read_text(encoding="utf-8"))
            selections = json.loads(Path(args.selections).read_text(encoding="utf-8")) if args.selections else None
            result = compare_experiment(plan, root=args.root, selections=selections,
                                        output=args.output, registration=registration)
        print(json.dumps({key: result[key] for key in ("experiment_id", "status", "comparison_status", "compatibility_issues") if key in result},
                         ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "invalid_or_unavailable", "error": str(exc)}, ensure_ascii=False))
        return 2
