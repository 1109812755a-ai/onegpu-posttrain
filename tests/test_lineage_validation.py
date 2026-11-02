import io
import json
from pathlib import Path

from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.build import build_finemath_artifacts
from onegpu_posttrain.data.finemath import FineMathBuildConfig
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage import (
    ArtifactRef,
    Digest,
    EvaluationLineage,
    ModelLineage,
    RunRecord,
    new_run_id,
    sha256_file,
)
from onegpu_posttrain.lineage_validation import validate_lineage_root


def _run(path, *, run_id=None, input_ref=None, incomplete=False):
    payload = RunRecord(
        run_id=run_id or new_run_id("train", label="fixture"), run_type="train", status="completed",
        lineage_status="incomplete" if incomplete else "complete", started_at="2026-10-17T00:00:00Z",
        recipe_path="recipes/fixture.yaml", resolved_config_sha256="a" * 64,
        inputs=[input_ref] if input_ref else [],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return payload


def test_plain_jsonl_is_allowed_but_not_complete_lineage(tmp_path, capsys):
    dataset = tmp_path / "plain.jsonl"
    dataset.write_text('{"messages":[]}', encoding="utf-8")
    run_path = tmp_path / "runs" / "one" / "lineage.json"
    _run(run_path, input_ref=ArtifactRef(
        id="file-plain", kind="file", path=str(dataset), digest=Digest(value=sha256_file(dataset), status="available"),
    ), incomplete=True)

    args = build_parser().parse_args(["validate-lineage", "--root", str(tmp_path)])
    assert args.func(args) == 0
    capsys.readouterr()
    result = validate_lineage_root(tmp_path, require_complete=True)
    assert not result["valid"]
    assert any(item["code"] == "incomplete" for item in result["errors"])


def test_validator_detects_duplicate_run_id_and_digest_mismatch(tmp_path):
    data = tmp_path / "dataset.jsonl"
    data.write_text("before", encoding="utf-8")
    run_id = new_run_id("train", label="duplicate")
    ref = ArtifactRef(id="dataset-a", kind="file", path=str(data), digest=Digest(value=sha256_file(data), status="available"))
    _run(tmp_path / "runs/a/lineage.json", run_id=run_id, input_ref=ref)
    _run(tmp_path / "runs/b/lineage.json", run_id=run_id, input_ref=ref)
    data.write_text("tampered", encoding="utf-8")

    result = validate_lineage_root(tmp_path)
    codes = {item["code"] for item in result["errors"]}
    assert {"duplicate_run_id", "digest_mismatch"} <= codes


def test_validator_detects_dangling_local_reference(tmp_path):
    missing = tmp_path / "runs" / "missing-output.jsonl"
    _run(tmp_path / "runs/one/lineage.json", input_ref=ArtifactRef(
        id="file-missing", kind="file", path=str(missing),
    ))

    result = validate_lineage_root(tmp_path)
    assert any(item["code"] == "dangling_local_ref" for item in result["errors"])


def test_validator_resolves_repository_relative_reference_from_nested_run_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    output = project / "runs" / "mini" / "one" / "merged" / "model.bin"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"model")
    nested_root = project / "runs" / "mini" / "one"
    _run(nested_root / "merge_runs" / "merge-one" / "lineage.json", input_ref=ArtifactRef(
        id="model-one", kind="model", path="runs/mini/one/merged/model.bin",
        digest=Digest(value=sha256_file(output), status="available"),
    ))

    result = validate_lineage_root(nested_root, require_complete=True)

    assert result["valid"], result["errors"]


def test_repository_root_scan_ignores_top_level_pytest_scratch_directories(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _run(project / "runs" / "one" / "lineage.json")

    scratch_lineage = project / ".pytest-tmp-full" / "runs" / "broken" / "lineage.json"
    scratch_lineage.parent.mkdir(parents=True)
    scratch_lineage.write_text("{not-json", encoding="utf-8")
    codex_scratch = project / ".tmp-pytest-lan-ui" / "runs" / "broken" / "lineage.json"
    codex_scratch.parent.mkdir(parents=True)
    codex_scratch.write_text("{not-json", encoding="utf-8")

    result = validate_lineage_root(project, require_complete=True)

    assert result["valid"], result["errors"]
    assert result["summary"]["runs"] == 1


def test_repository_root_scan_prunes_tool_install_trees_before_descent(tmp_path, monkeypatch):
    import onegpu_posttrain.lineage_validation as validation

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _run(project / "runs" / "one" / "lineage.json")
    for excluded in (".git", ".venv", "web/node_modules"):
        invalid = project / excluded / "deep" / "lineage.json"
        invalid.parent.mkdir(parents=True)
        invalid.write_text("{not-json", encoding="utf-8")

    visited = []
    walk_calls = 0
    original_walk = validation.os.walk

    def recording_walk(*args, **kwargs):
        nonlocal walk_calls
        walk_calls += 1
        for current, directories, files in original_walk(*args, **kwargs):
            visited.append(Path(current))
            yield current, directories, files

    monkeypatch.setattr(validation.os, "walk", recording_walk)
    result = validate_lineage_root(project, require_complete=True)

    assert result["valid"], result["errors"]
    assert result["summary"]["runs"] == 1
    assert walk_calls == 1
    assert all(not ({".git", ".venv", "node_modules"} & set(path.relative_to(project).parts))
               for path in visited)


def test_validator_detects_model_lineage_cycle(tmp_path):
    directory = tmp_path / "run" / "model_artifacts"
    directory.mkdir(parents=True)
    first = ModelLineage(model_id="model-a", kind="merged", digest=Digest(status="unavailable", reason="fixture"), parent_model_ids=["model-b"])
    second = ModelLineage(model_id="model-b", kind="merged", digest=Digest(status="unavailable", reason="fixture"), parent_model_ids=["model-a"])
    (directory / "a.json").write_text(first.model_dump_json(), encoding="utf-8")
    (directory / "b.json").write_text(second.model_dump_json(), encoding="utf-8")

    result = validate_lineage_root(tmp_path)
    assert any(item["code"] == "lineage_cycle" for item in result["errors"])


def test_validator_warns_for_external_unavailable_evidence(tmp_path):
    run_path = tmp_path / "runs" / "remote" / "lineage.json"
    _run(run_path, input_ref=ArtifactRef(
        id="model-remote", kind="model", path="org/model", digest=Digest(status="unavailable", reason="remote model"),
    ))

    result = validate_lineage_root(tmp_path)
    assert result["valid"]
    assert any(item["code"] == "unavailable_evidence" for item in result["warnings"])


def test_validator_accepts_complete_built_dataset_fixture(tmp_path):
    rows = [
        {"url": f"fixture:{index}", "text": f"A mathematical statement with unique number {index}. " * 8}
        for index in range(12)
    ]
    build_finemath_artifacts(
        rows, output_dir=tmp_path, tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(train_token_budget=20, holdout_token_budget=10,
                                   max_seq_tokens=10, holdout_permille=500),
        source_repo="fixture", source_config=None, source_revision="rev-1",
        tokenizer_repo="fixture-tokenizer", tokenizer_revision="rev-1",
    )

    result = validate_lineage_root(tmp_path, require_complete=True)
    assert result["valid"], result["errors"]


def test_validator_closes_each_training_dataset_id_and_digest(tmp_path):
    origin = DataSource(source_id="fixture", kind="local_upload", name="fixture", license_status="not_provided", declared_by="user")
    datasets = []
    for index in range(2):
        row = {"id": f"sft-{index}", "raw_problem": f"Question {index}?", "messages": [
            {"role": "user", "content": f"Question {index}?"}, {"role": "assistant", "content": f"Answer {index}."}],
            "prompt_template_id": "chat-v1"}
        raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        datasets.append(register_jsonl(io.BytesIO(raw), workspace=tmp_path,
            output_root=tmp_path / "data" / "processed" / f"dataset-{index}", stage="sft", origin=origin))

    run = RunRecord(run_id=new_run_id("train", label="two-datasets"), run_type="train", status="completed",
        started_at="2026-10-24T00:00:00Z", recipe_path="recipes/fixture.yaml", resolved_config_sha256="a" * 64,
        inputs=[ArtifactRef(
            id=item.dataset_id, kind="dataset", path=str(item.path / "canonical" / "sft.jsonl"),
            digest=Digest(value=item.canonical_sha256, status="available"), role="train") for item in datasets])
    lineage = tmp_path / "runs" / run.run_id / "lineage.json"
    lineage.parent.mkdir(parents=True)
    lineage.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    result = validate_lineage_root(tmp_path, require_complete=True)
    assert result["valid"], result["errors"]
    assert result["summary"]["datasets"] == 2
    assert len(run.inputs) == 2

    broken = run.model_copy(update={"inputs": [run.inputs[0], run.inputs[1].model_copy(update={
        "digest": Digest(value="f" * 64, status="available")})]})
    lineage.write_text(broken.model_dump_json(indent=2), encoding="utf-8")
    result = validate_lineage_root(tmp_path)
    assert any(item["code"] == "dataset_input_digest_mismatch" for item in result["errors"])


def test_validator_checks_model_producer_and_model_content_digest(tmp_path):
    model_dir = tmp_path / "models" / "model-output"
    model_dir.mkdir(parents=True)
    weights = model_dir / "weights.bin"
    weights.write_bytes(b"fixture weights")
    digest = sha256_file(weights)
    model = ModelLineage(model_id="model-output", kind="adapter", path=str(weights),
        digest=Digest(value=digest, status="available"), produced_by_run_id="run-train-output", stage="sft")
    artifact = tmp_path / "runs" / "run-train-output" / "model_artifacts" / "model-output.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(model.model_dump_json(), encoding="utf-8")
    run = RunRecord(run_id="run-train-output", run_type="train", status="completed",
        started_at="2026-10-24T00:00:00Z", recipe_path="recipes/fixture.yaml", resolved_config_sha256="a" * 64,
        outputs=[ArtifactRef(id=model.model_id, kind="model", path=str(weights),
        digest=Digest(value=digest, status="available"), role="adapter")])
    run_path = artifact.parent.parent / "lineage.json"
    run_path.write_text(run.model_dump_json(), encoding="utf-8")
    assert validate_lineage_root(tmp_path, require_complete=True)["valid"]

    model_dir.joinpath("weights.bin").write_bytes(b"tampered weights")
    result = validate_lineage_root(tmp_path)
    assert any(item["code"] == "model_digest_mismatch" for item in result["errors"])


def test_validator_closes_cpt_sft_fanout_and_evaluation_model_digest(tmp_path):
    source = DataSource(source_id="fixture", kind="local_upload", name="fixture", license_status="not_provided", declared_by="user")
    datasets = []
    for index in range(2):
        row = {"id": f"sft-branch-{index}", "raw_problem": f"Question {index}?", "messages": [
            {"role": "user", "content": f"Question {index}?"}, {"role": "assistant", "content": f"Answer {index}."}],
            "prompt_template_id": "chat-v1"}
        datasets.append(register_jsonl(io.BytesIO((json.dumps(row) + "\n").encode()), workspace=tmp_path,
            output_root=tmp_path / "data" / "processed" / f"branch-{index}", stage="sft", origin=source))

    def write_model(model_id, *, stage, kind, parent_ids=(), producer=None):
        weights = tmp_path / "models" / model_id / "weights.bin"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(model_id.encode())
        model = ModelLineage(model_id=model_id, kind=kind, stage=stage, path=str(weights),
            digest=Digest(value=sha256_file(weights), status="available"), parent_model_ids=list(parent_ids),
            produced_by_run_id=producer)
        registry = tmp_path / "runs" / (producer or "external") / "model_artifacts" / f"{model_id}.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(model.model_dump_json(), encoding="utf-8")
        return model, ArtifactRef(id=model_id, kind="model", path=str(weights),
            digest=model.digest, role="parent")

    base, base_ref = write_model("model-base", stage="base", kind="base")
    cpt_run_id = "run-cpt-chain"
    cpt, cpt_output = write_model("model-cpt-chain", stage="cpt", kind="merged", parent_ids=[base.model_id], producer=cpt_run_id)
    cpt_run = RunRecord(run_id=cpt_run_id, run_type="train", status="completed", started_at="2026-10-24T00:00:00Z",
        recipe_path="recipes/cpt.yaml", resolved_config_sha256="b" * 64, inputs=[base_ref], outputs=[cpt_output])
    cpt_lineage = tmp_path / "runs" / cpt_run_id / "lineage.json"
    cpt_lineage.parent.mkdir(parents=True, exist_ok=True)
    cpt_lineage.write_text(cpt_run.model_dump_json(), encoding="utf-8")

    sft_run_id = "run-sft-chain"
    sft, sft_output = write_model("model-sft-chain", stage="sft", kind="merged", parent_ids=[cpt.model_id], producer=sft_run_id)
    dataset_refs = [ArtifactRef(id=item.dataset_id, kind="dataset", path=str(item.path / "canonical" / "sft.jsonl"),
        digest=Digest(value=item.canonical_sha256, status="available"), role="train") for item in datasets]
    sft_run = RunRecord(run_id=sft_run_id, run_type="train", status="completed", started_at="2026-10-24T00:00:00Z",
        recipe_path="recipes/sft.yaml", resolved_config_sha256="c" * 64, inputs=[cpt_ref := ArtifactRef(
            id=cpt.model_id, kind="model", path=cpt.path, digest=cpt.digest, role="parent"), *dataset_refs], outputs=[sft_output])
    sft_lineage = tmp_path / "runs" / sft_run_id / "lineage.json"
    sft_lineage.parent.mkdir(parents=True, exist_ok=True)
    sft_lineage.write_text(sft_run.model_dump_json(), encoding="utf-8")

    branch_runs = {}
    for stage in ("dpo", "simpo", "grpo"):
        run_id = f"run-{stage}-branch"
        model, output = write_model(f"model-{stage}-branch", stage=stage, kind="merged",
            parent_ids=[sft.model_id], producer=run_id)
        branch = RunRecord(run_id=run_id, run_type="train", status="completed", started_at="2026-10-24T00:00:00Z",
            recipe_path=f"recipes/{stage}.yaml", resolved_config_sha256="d" * 64,
            inputs=[ArtifactRef(id=sft.model_id, kind="model", path=sft.path, digest=sft.digest, role="parent")],
            outputs=[output])
        branch_lineage = tmp_path / "runs" / run_id / "lineage.json"
        branch_lineage.parent.mkdir(parents=True, exist_ok=True)
        branch_lineage.write_text(branch.model_dump_json(), encoding="utf-8")
        branch_runs[stage] = (model, branch)

    grpo_model, grpo_run = branch_runs["grpo"]
    eval_run_id = "run-eval-grpo-branch"
    eval_run = RunRecord(run_id=eval_run_id, run_type="evaluate", status="completed", started_at="2026-10-24T00:00:00Z",
        inputs=[ArtifactRef(id=grpo_model.model_id, kind="model", path=grpo_model.path, digest=grpo_model.digest, role="evaluated"),
                ArtifactRef(id="benchmark-math", kind="dataset", role="benchmark", availability="unavailable")])
    evaluation_dir = tmp_path / "runs" / eval_run_id
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "lineage.json").write_text(eval_run.model_dump_json(), encoding="utf-8")
    evaluation = EvaluationLineage(eval_id="eval-grpo-branch", produced_by_run_id=eval_run_id,
        model_id=grpo_model.model_id, model_produced_by_run_id=grpo_run.run_id,
        dataset_ids=["benchmark-math"], protocol_version="fixture-v1", status="completed")
    (evaluation_dir / "evaluation_lineage.json").write_text(evaluation.model_dump_json(), encoding="utf-8")

    result = validate_lineage_root(tmp_path, require_complete=True)
    assert result["valid"], result["errors"]
    assert {"cpt", "sft", "dpo", "simpo", "grpo"} <= {model.stage for _, model in [
        (None, cpt), (None, sft), *[(None, item[0]) for item in branch_runs.values()]]}

    broken_eval = eval_run.model_copy(update={"inputs": [eval_run.inputs[0].model_copy(update={
        "digest": Digest(value="e" * 64, status="available")}), eval_run.inputs[1]]})
    (evaluation_dir / "lineage.json").write_text(broken_eval.model_dump_json(), encoding="utf-8")
    result = validate_lineage_root(tmp_path)
    assert any(item["code"] == "evaluation_model_digest_mismatch" for item in result["errors"])

