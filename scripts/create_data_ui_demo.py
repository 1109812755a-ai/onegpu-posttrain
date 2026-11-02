"""Explicitly generate synthetic Data UI examples; never modify real Smoke artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from onegpu_posttrain.data.build import build_finemath_artifacts, build_openr1_artifacts
from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.finemath import FineMathBuildConfig
from onegpu_posttrain.data.openr1 import OpenR1BuildConfig
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer


def openr1_rows():
    for i in range(350):
        if i % 9 == 0:
            yield {"problem": ""}
        row = {
            "problem": f"Synthetic example {i}: solve x + {i} = {i + 2}.",
            "answer": "2", "source": "synthetic-algebra" if i % 3 else "synthetic-geometry",
            "problem_type": "fixture", "question_type": "math-word-problem", "uuid": f"synthetic-{i}",
            "generations": [f"Subtract the constant {i}. The answer is \\boxed{{2}}.",
                            f"Add the constant {i}. The answer is \\boxed{{4}}.", "Unfinished reasoning"],
            "is_reasoning_complete": [True, True, False], "finish_reasons": ["stop", "stop", "length"],
            "correctness_math_verify": [True, False, False],
        }
        if i % 7 == 0:
            row["correctness_math_verify"] = [False, False, False]
        if i % 11 == 0:
            row["generations"][0] = "long reasoning " * 90 + r"\boxed{2}"
        if i % 13 == 0:
            row["generations"][1] = "Complete text without an answer marker"
        yield row
        if i % 8 == 0:
            yield row


def finemath_rows():
    for i in range(300):
        yield {"url": f"fixture://finemath/{i}", "text": "" if i % 9 == 0 else f"Synthetic mathematics document {i}. " + "algebra theorem proof reasoning " * 35,
               "score": 4.0, "int_score": 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui-demo"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output is not empty; choose a fresh directory. Existing artifacts were left intact.")
    common = {"tokenizer": WhitespaceTokenizer(), "source_config": "synthetic-only", "source_revision": "fixture-v1",
              "tokenizer_repo": "whitespace-test-only", "tokenizer_revision": "fixture-v1"}
    openr1 = build_openr1_artifacts(
        openr1_rows(), output_dir=args.output / "openr1-demo", source_repo="onegpu/demo-openr1",
        config=OpenR1BuildConfig(sft_target=24, alignment_target=30, holdout_target=12, preference_target=12,
                                preference_eval_target=6, grpo_target=18, sft_max_length=80, preference_max_length=80,
                                grpo_max_prompt_length=50, max_raw_rows=400),
        benchmark_index=BenchmarkIndex([("demo-only", "Synthetic example 1: solve x + 1 = 3.")]), **common,
    )
    finemath = build_finemath_artifacts(
        finemath_rows(), output_dir=args.output / "finemath-demo", source_repo="onegpu/demo-finemath",
        config=FineMathBuildConfig(train_token_budget=2500, holdout_token_budget=500, max_seq_tokens=64,
                                  holdout_permille=250, max_raw_rows=300), **common,
    )
    print(f"Synthetic examples only (not training results): {args.output.resolve()}")
    print(f"OpenR1: {openr1['stop_reason']}; FineMath: {finemath['stop_reason']}")


if __name__ == "__main__":
    main()
