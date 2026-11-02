from pathlib import Path

from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.policy import (
    DataPolicy,
    PolicyTokenizer,
    SourceDefinition,
    SplitSnapshot,
)


def policy_fixture(root: Path, *, stage="sft", target=3, max_length=64):
    root.mkdir(parents=True, exist_ok=True)
    rows = [{"problem": f"Calculate the integer {i} plus 2.", "answer": str(i + 2),
             "uuid": f"own-{i}", "source": "project-owned-fixture",
             "generations": [f"Addition gives \\boxed{{{i + 2}}}.", f"Wrong addition gives \\boxed{{{i + 3}}}."],
             "is_reasoning_complete": [True, True], "finish_reasons": ["stop", "stop"],
             "correctness_math_verify": [True, False]} for i in range(100)]
    write_jsonl(root / "raw.jsonl", rows)
    write_jsonl(root / "benchmark.jsonl", [{"problem": "Describe the evaluation-only geography of the moon Neptune."}])
    tokenizer = PolicyTokenizer(test_only=True)
    source = SourceDefinition(title="项目自有测试", input_jsonl=str(root / "raw.jsonl"),
                              source_repo="project-owned-fixture", source_revision="v1",
                              benchmark_jsonl=str(root / "benchmark.jsonl"), license_status="declared",
                              license_note="Project-owned test data", tokenizer=tokenizer,
                              split=SplitSnapshot(sft_weight=1 if stage == "sft" else 0,
                                                  alignment_weight=1 if stage == "preference" else 0,
                                                  holdout_weight=1))
    policy = DataPolicy(stage=stage, tokenizer=tokenizer,
                        resources={"max_length": max_length, "target_count": target})
    return policy, source, rows
