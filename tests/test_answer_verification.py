from __future__ import annotations

import pytest

from onegpu_posttrain.evaluation import verification
from onegpu_posttrain.evaluation.answers import answers_equal
from onegpu_posttrain.synthetic.verification import assess_candidate_answers_batch


def test_exact_match_is_verified_without_starting_a_worker(monkeypatch):
    monkeypatch.setattr(verification, "_run_bounded_batch", lambda *_: pytest.fail("worker must not start"))
    result = verification.verify_answer_pairs([(r"\boxed{1,024}", "1024")])[0]
    assert result.status == "correct"
    assert result.method == "normalize-answer:v1"


def test_nonliteral_outcomes_are_preserved_from_the_bounded_worker(monkeypatch):
    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "0.9.0"))
    monkeypatch.setattr(verification, "_run_bounded_batch", lambda pairs, timeout: [
        "correct", "incorrect", "timeout", "verifier_error",
    ])
    results = verification.verify_answer_pairs([
        ("1/2", "0.5"), ("3", "4"), ("bad", "5"), ("x", "y"),
    ])
    assert [item.status for item in results] == ["correct", "incorrect", "timeout", "verifier_error"]
    assert all(item.verifier_version == "0.9.0" for item in results)


def test_missing_optional_verifier_is_not_a_negative_result(monkeypatch):
    monkeypatch.setattr(verification, "_verification_identity", lambda: (None, None))
    monkeypatch.setattr(verification, "_run_bounded_batch", lambda *_: pytest.fail("worker must not start"))
    result = verification.verify_answer_pairs([("1/2", "0.5")])[0]
    assert result.status == "unverified"


def test_batch_size_bounds_worker_payloads(monkeypatch):
    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "test"))
    seen = []

    def worker(pairs, timeout):
        seen.append(len(pairs))
        return ["correct"] * len(pairs)

    monkeypatch.setattr(verification, "_run_bounded_batch", worker)
    results = verification.verify_answer_pairs([(f"x+{i}", f"y+{i}") for i in range(5)], batch_size=2)
    assert seen == [2, 2, 1]
    assert len(results) == 5
    assert all(result.status == "correct" for result in results)


def test_bounded_batch_timeout_terminates_and_joins_worker(monkeypatch):
    class Connection:
        def __init__(self, *, ready):
            self.ready = ready
            self.closed = False

        def poll(self, _timeout):
            return self.ready

        def close(self):
            self.closed = True

    class Process:
        pid = 1
        exitcode = None

        def __init__(self):
            self.alive = False
            self.terminated = False
            self.joined = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            self.joined = True

        def kill(self):
            self.alive = False

    receive, send, process = Connection(ready=False), Connection(ready=False), Process()

    class Context:
        @staticmethod
        def Pipe(duplex=False):
            return receive, send

        @staticmethod
        def Process(target, args):
            return process

    monkeypatch.setattr(verification.multiprocessing, "get_context", lambda _method: Context())
    result = verification._run_bounded_batch([verification.AnswerPair(prediction="1/2", reference="0.5")], 0.01)
    assert result == ["timeout"]
    assert process.terminated and process.joined
    assert receive.closed and send.closed


def test_bounded_batch_worker_eof_is_reported_as_verifier_error(monkeypatch):
    class Connection:
        def __init__(self, *, ready):
            self.ready = ready

        def poll(self, _timeout):
            return self.ready

        def recv(self):
            raise EOFError

        def close(self):
            pass

    class Process:
        pid = 1
        exitcode = 1

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

    receive, send = Connection(ready=True), Connection(ready=False)

    class Context:
        @staticmethod
        def Pipe(duplex=False):
            return receive, send

        @staticmethod
        def Process(target, args):
            return Process()

    monkeypatch.setattr(verification.multiprocessing, "get_context", lambda _method: Context())
    result = verification._run_bounded_batch([verification.AnswerPair(prediction="x", reference="y")], 1)
    assert result == ["verifier_error"]


def test_real_math_verify_batch_uses_a_single_bounded_worker():
    import importlib.util

    if importlib.util.find_spec("math_verify") is None:
        pytest.skip("optional math-verify dependency is unavailable")
    results = verification.verify_answer_pairs(
        [("2/4", "1/2"), ("3", "4")], batch_size=2, timeout_seconds=45,
    )
    assert [result.status for result in results] == ["correct", "incorrect"]


@pytest.mark.parametrize("batch_size,timeout", [(0, 1), (1, 0), (-1, 2), (1, -1)])
def test_verification_rejects_unbounded_or_empty_settings(batch_size, timeout):
    with pytest.raises(ValueError, match="must be positive"):
        verification.verify_answer_pairs([], batch_size=batch_size, timeout_seconds=timeout)


def test_boolean_compatibility_wrapper_returns_true_only_for_verified_correct(monkeypatch):
    monkeypatch.setattr(verification, "verify_answer_pairs", lambda pairs: [
        verification.AnswerVerificationResult(status="timeout", method="test"),
    ])
    assert answers_equal("1/2", "0.5") is False


def test_assessment_keeps_worker_failure_distinct_from_incorrect():
    assessment = assess_candidate_answers_batch(
        ["Final answer: 1/2"], ["0.5"],
        equivalence_verifier=lambda *_: (_ for _ in ()).throw(OSError("worker failure")),
    )[0]
    assert assessment.status == "verifier_error"
    assert assessment.status != "incorrect"
