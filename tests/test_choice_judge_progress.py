import json
import time
from datetime import datetime
from types import SimpleNamespace

from onegpu_posttrain.evaluation import choice_judge
from onegpu_posttrain.synthetic.providers import ProviderError
from onegpu_posttrain.ui.launcher import SFTLauncher


def test_slow_fake_judge_emits_safe_live_progress_readable_by_launcher(tmp_path, monkeypatch):
    records = [
        {"id": "question-1", "problem_id": "p1", "raw_problem": "private-question-secret",
         "prompt": "private-question-secret", "answer": "A", "eval_prompt_version": "fixture-v1",
         "question_type": "single_choice", "options": ["first", "second"]},
        {"id": "question-2", "problem_id": "p2", "raw_problem": "private-question-secret",
         "prompt": "private-question-secret", "answer": "B", "eval_prompt_version": "fixture-v1",
         "question_type": "single_choice", "options": ["first", "second"]},
        {"id": "question-3", "problem_id": "p3", "raw_problem": "private-question-secret",
         "prompt": "private-question-secret", "answer": "A", "eval_prompt_version": "fixture-v1",
         "question_type": "single_choice", "options": ["first", "second"]},
    ]
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    output = tmp_path / "runs" / "evaluations" / "run-live-choice-fixture" / "choice_accuracy"
    workspace = tmp_path
    profile = SimpleNamespace(profile_id="fixture-profile", api_key_env="FIXTURE_KEY",
                              model="fixture-model", revision=None)

    monkeypatch.setattr(choice_judge, "load_choice_judge_profile", lambda _workspace: profile)
    monkeypatch.setattr(choice_judge, "require_provider_credentials", lambda _profile: None)
    monkeypatch.setattr("onegpu_posttrain.synthetic.secrets.load_local_provider_credential", lambda *_args: None)

    class FakeGenerator:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, _prompt):
            return "private-model-output"

    class FakeJudge:
        def __init__(self, _profile):
            pass

        def judge(self, record, _raw_output):
            time.sleep(0.01)
            if record.id == "question-1":
                return choice_judge.ChoiceJudgeDecision(outcome="correct", rationale="match"), {
                    "prompt_tokens": None, "completion_tokens": None,
                }
            if record.id == "question-2":
                return choice_judge.ChoiceJudgeDecision(outcome="undetermined", rationale="unclear"), {
                    "prompt_tokens": None, "completion_tokens": None,
                }
            raise ProviderError("transport", "sanitized", retryable=False, uncertain=True)

    from onegpu_posttrain.evaluation import internal_math

    monkeypatch.setattr(internal_math, "TransformersGenerator", FakeGenerator)
    monkeypatch.setattr(choice_judge, "BailianChoiceJudge", FakeJudge)

    run_id = "run-live-choice-fixture"
    log_path = tmp_path / ".onegpu-posttrain" / "launcher" / "logs" / f"{run_id}.eval.log"
    log_path.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    launcher = SFTLauncher(workspace)
    launcher._write_state({
        "run_id": run_id, "run_kind": "evaluation", "status": "running",
        "evaluation_status": "running", "scoring_strategy": "bailian_llm_judge",
        "evaluation_sample_count": len(records), "output": output.parent.relative_to(workspace).as_posix(),
        "log_path": log_path.relative_to(workspace).as_posix(),
        "started_at": datetime.now().astimezone().isoformat(),
    })

    class ActiveProcess:
        @staticmethod
        def poll():
            return None

    launcher._process = ActiveProcess()
    snapshots = []

    def progress_callback(event):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(choice_judge.choice_judge_progress_line(event) + "\n")
            stream.flush()
        observed = launcher.get_run(run_id)
        snapshots.append(observed["choice_judge_progress"])

    result = choice_judge.evaluate_registered_choice_dataset_with_judge(
        model_path=model_path, canonical_path=canonical, output_dir=output,
        workspace=workspace, progress_callback=progress_callback,
    )

    assert any(item["phase"] == "bailian_judging" for item in snapshots)
    recorded = [item for item in snapshots if item["phase"] == "recorded"]
    assert [item["completed"] for item in recorded] == [1, 2, 3]
    assert recorded[-1]["counts"] == {
        "correct": 1, "incorrect": 0, "undetermined": 1, "provider_errors": 1,
    }
    assert recorded[-1]["events"][-1]["error_category"] == "transport"
    assert result["evaluated_count"] == 3
    progress_log = log_path.read_text(encoding="utf-8")
    assert "private-question-secret" not in progress_log
    assert "private-model-output" not in progress_log
    assert "question-1" not in progress_log


def test_choice_judge_progress_parser_ignores_untrusted_event_fields():
    safe = {
        "completed": 1, "total": 2,
        "counts": {"correct": 1, "incorrect": 0, "undetermined": 0, "provider_errors": 0},
        "phase": "recorded", "current_question": 1,
    }
    unsafe = {**safe, "question": "private-question", "model_output": "private-output"}
    text = choice_judge.choice_judge_progress_line(safe) + "\n"
    text += choice_judge.CHOICE_JUDGE_PROGRESS_PREFIX + json.dumps(unsafe) + "\n"

    progress = choice_judge.parse_choice_judge_progress(text, total=2)

    assert progress["completed"] == 1
    assert progress["phase"] == "recorded"
    assert set(progress["events"][-1]) == {
        "completed", "total", "counts", "phase", "current_question", "error_category",
    }
