from onegpu_posttrain.data.capabilities import check_source_stage, source_stage_capabilities


def test_source_stage_capability_matrix_matches_mvp_support():
    entries = {item.source_id: item for item in source_stage_capabilities()}
    assert entries["upload"].supported_stages == ["cpt", "sft", "preference", "grpo"]
    assert entries["finemath"].supported_stages == ["cpt"]
    assert entries["deepmath"].supported_stages == ["sft"]
    assert entries["openr1"].supported_stages == ["sft", "preference", "grpo"]
    assert entries["llm-generation"].supported_stages == ["sft", "preference"]
    assert entries["llm-generation-cpt"].status == "unsupported"
    assert entries["llm-generation-grpo"].status == "unsupported"


def test_unsupported_and_unknown_source_stage_pairs_return_explanation():
    assert check_source_stage("openr1", "cpt")[0] is False
    assert "不支持" in check_source_stage("openr1", "cpt")[1]
    assert "CPT" in check_source_stage("llm-generation-cpt", "cpt")[1]
    assert check_source_stage("not-a-source", "sft")[0] is False
    assert check_source_stage("upload", "preference")[0] is True
