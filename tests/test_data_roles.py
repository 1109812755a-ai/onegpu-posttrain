from __future__ import annotations

import pytest
from pydantic import ValidationError

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.roles import (
    check_generated_candidate,
    load_role_registry,
    register_dataset,
    save_role_registry,
    select_training_seeds,
)
from onegpu_posttrain.data.targeted import (
    DataRoleRegistry,
    RoleRecord,
    ScenarioDefinition,
    ScenarioLabel,
)
from onegpu_posttrain.lineage import sha256_file

SHA = "a" * 64


def test_registry_rejects_downgrade_missing_parent_and_cycles():
    root = RoleRecord(dataset_id="train", dataset_sha256=SHA, record_id="train-1", problem_id="p1", role="train_seed")
    diagnostic = RoleRecord(dataset_id="dev", dataset_sha256=SHA, record_id="dev-1", problem_id="p2", role="diagnostic_dev")
    final = RoleRecord(dataset_id="holdout", dataset_sha256=SHA, record_id="hold-1", problem_id="p3", role="final_holdout")
    with pytest.raises(ValidationError, match="降级"):
        DataRoleRegistry(records=[root, diagnostic, RoleRecord(dataset_id="derived", dataset_sha256=SHA, record_id="candidate-1",
                                                              problem_id="p4", role="train_seed",
                                                              parent_record_ids=["dev-1"])])
    with pytest.raises(ValidationError, match="缺少派生记录"):
        DataRoleRegistry(records=[RoleRecord(dataset_id="derived", dataset_sha256=SHA, record_id="candidate-1", problem_id="p4",
                                             role="train_seed", parent_record_ids=["missing"])])
    with pytest.raises(ValidationError, match="循环"):
        DataRoleRegistry(records=[
            RoleRecord(dataset_id="d", dataset_sha256=SHA, record_id="a", problem_id="p1", role="train_seed", parent_record_ids=["b"]),
            RoleRecord(dataset_id="d", dataset_sha256=SHA, record_id="b", problem_id="p2", role="train_seed", parent_record_ids=["a"]),
        ])
    DataRoleRegistry(records=[root, diagnostic, final])


def test_registry_extend_creates_immutable_version_and_roundtrips(tmp_path):
    initial = DataRoleRegistry(records=[RoleRecord(dataset_id="train", dataset_sha256=SHA, record_id="train-1", problem_id="p1", role="train_seed")])
    prior_hash = initial.registry_sha256
    extended = initial.extend([RoleRecord(dataset_id="train", dataset_sha256=SHA, record_id="train-2", problem_id="p2", role="train_seed")])
    assert initial.version == 1 and len(initial.records) == 1 and initial.registry_sha256 == prior_hash
    assert extended.version == 2 and len(extended.records) == 2 and extended.registry_sha256 != prior_hash
    path = tmp_path / "role-registry.json"
    save_role_registry(extended, path)
    assert load_role_registry(path) == extended
    with pytest.raises(FileExistsError):
        save_role_registry(extended, path)


def test_seed_selection_requires_roles_and_protected_index_and_reports_shortfall(tmp_path):
    dataset = tmp_path / "train.jsonl"
    protected_text = "Find the sum of 19 and 7 in the triangle question."
    rows = [
        {"id": "s1", "problem_id": "p1", "raw_problem": "A unique algebra problem with x and y.", "metadata": {"topic": "algebra"}},
        {"id": "s2", "problem_id": "p2", "raw_problem": "A unique geometry problem about a circle.", "metadata": {"topic": "geometry"}},
        {"id": "s3", "problem_id": "p3", "raw_problem": protected_text, "metadata": {"topic": "algebra"}},
        {"id": "s4", "problem_id": "protected-pid", "raw_problem": "Different wording but protected problem identity.", "metadata": {"topic": "algebra"}},
    ]
    write_jsonl(dataset, rows)
    registry = register_dataset(None, dataset, dataset_id="train-math", role="train_seed")
    registry = registry.extend([
        RoleRecord(dataset_id="diag", dataset_sha256=SHA, record_id="d1", problem_id="p-dev", role="diagnostic_dev"),
        RoleRecord(dataset_id="holdout", dataset_sha256=SHA, record_id="h1", problem_id="protected-pid", role="final_holdout"),
    ])
    definition = ScenarioDefinition(
        taxonomy_id="math", taxonomy_version="v1", dataset_id="eval-dev", dataset_sha256=SHA,
        label_source="metadata", label_field="topic",
        scenarios=[ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"])],
    )
    before = sha256_file(dataset)
    index = BenchmarkIndex([("holdout:1", protected_text)])
    with pytest.raises(ValueError, match="缺少.*污染索引"):
        select_training_seeds(dataset, dataset_id="train-math", registry=registry, definition=definition,
                              scenario_id="algebra", target_count=5, seed=7, protected_index=None,
                              protected_index_id=None, protected_index_sha256=None)

    result = select_training_seeds(dataset, dataset_id="train-math", registry=registry, definition=definition,
                                   scenario_id="algebra", target_count=5, seed=7, protected_index=index,
                                   protected_index_id="holdout-v1", protected_index_sha256=SHA)
    again = select_training_seeds(dataset, dataset_id="train-math", registry=registry, definition=definition,
                                  scenario_id="algebra", target_count=5, seed=7, protected_index=index,
                                  protected_index_id="holdout-v1", protected_index_sha256=SHA)
    assert [row["id"] for row in result.records] == ["s1"]
    assert [row["id"] for row in again.records] == ["s1"]
    assert result.report["shortfall"] == 4
    assert result.report["rejected"] == {
        "scenario_label_mismatch_or_unknown": 1,
        "contamination": 1,
        "protected_problem_id_overlap": 1,
    }
    assert result.report["contamination_index_checked"] is True
    assert result.report["contamination_index_id"] == "holdout-v1"
    assert result.report["contamination_index_sha256"] == SHA
    assert sha256_file(dataset) == before


def test_generated_candidate_requires_train_only_ancestry_and_clean_text():
    registry = DataRoleRegistry(records=[
        RoleRecord(dataset_id="train", dataset_sha256=SHA, record_id="seed", problem_id="p1", role="train_seed"),
        RoleRecord(dataset_id="holdout", dataset_sha256=SHA, record_id="holdout", problem_id="p2", role="final_holdout"),
    ])
    index = BenchmarkIndex([("holdout:1", "A private final holdout problem text here.")])
    leaked_ancestor = check_generated_candidate(
        {"problem": "A new generated math question", "metadata": {"parent_record_ids": ["holdout"]}},
        registry=registry, protected_index=index, protected_index_id="holdout-v1", protected_index_sha256=SHA,
    )
    assert not leaked_ancestor.accepted
    assert "protected_ancestor" in leaked_ancestor.reasons
    reused_problem_id = check_generated_candidate(
        {"problem_id": "p2", "raw_problem": "A unique new question.", "metadata": {"parent_record_ids": ["seed"]}},
        registry=registry, protected_index=index, protected_index_id="holdout-v1", protected_index_sha256=SHA,
    )
    assert not reused_problem_id.accepted
    assert "protected_problem_id_overlap" in reused_problem_id.reasons
    contaminated = check_generated_candidate(
        {"raw_problem": "A private final holdout problem text here.", "metadata": {"parent_record_ids": ["seed"]}},
        registry=registry, protected_index=index, protected_index_id="holdout-v1", protected_index_sha256=SHA,
    )
    assert not contaminated.accepted
    assert contaminated.contamination_hits[0]["kind"] == "exact"
    clean = check_generated_candidate(
        {"raw_problem": "A fresh, unrelated algebra question.", "metadata": {"parent_record_ids": ["seed"]}},
        registry=registry, protected_index=index, protected_index_id="holdout-v1", protected_index_sha256=SHA,
    )
    assert clean.accepted
