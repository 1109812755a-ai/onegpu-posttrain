import json

import pytest

from onegpu_posttrain.ui.model_display_names import (
    DisplayNameStoreError,
    ModelDisplayNameStore,
    validate_display_name,
)


def test_display_name_store_persists_updates_and_clear_by_immutable_model_id(tmp_path):
    store = ModelDisplayNameStore(tmp_path)
    registered = {"model-sft-a", "model-dpo-b"}

    assert store.set("model-sft-a", "  数学助手  ", registered_model_ids=registered) == "数学助手"
    store.set("model-dpo-b", "DPO 实验二", registered_model_ids=registered)
    assert store.get_all(registered) == {"model-sft-a": "数学助手", "model-dpo-b": "DPO 实验二"}

    store.clear("model-sft-a", registered_model_ids=registered)

    assert store.get_all(registered) == {"model-dpo-b": "DPO 实验二"}
    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted == {"schema": "model-display-names-v1", "display_names": {"model-dpo-b": "DPO 实验二"}}


@pytest.mark.parametrize("value", ["", "   ", "x" * 81, "name\nother", "name\x00bad", None, 42])
def test_display_name_validation_rejects_empty_long_or_control_input(value):
    with pytest.raises(ValueError):
        validate_display_name(value)


def test_display_name_store_rejects_unregistered_and_path_like_model_ids(tmp_path):
    store = ModelDisplayNameStore(tmp_path)

    with pytest.raises(KeyError, match="尚未登记"):
        store.set("model-unknown", "name", registered_model_ids={"model-known"})
    with pytest.raises(ValueError, match="model_id"):
        store.set("../model-known", "name", registered_model_ids={"../model-known"})
    assert not store.path.exists()


def test_display_name_store_ignores_valid_stale_keys_when_catalog_ids_are_supplied(tmp_path):
    store = ModelDisplayNameStore(tmp_path)
    store.set("model-current", "当前模型", registered_model_ids={"model-current"})
    store.set("model-removed", "旧模型", registered_model_ids={"model-current", "model-removed"})

    assert store.get_all({"model-current"}) == {"model-current": "当前模型"}
    assert store.get_all() == {"model-current": "当前模型", "model-removed": "旧模型"}


def test_display_name_store_backs_up_malformed_record_before_explicit_recovery(tmp_path):
    store = ModelDisplayNameStore(tmp_path)
    store.directory.mkdir(parents=True)
    store.path.write_text("{broken", encoding="utf-8")

    assert store.get_all() == {}
    store.set("model-good", "恢复后名称", registered_model_ids={"model-good"})

    backups = list(store.directory.glob("model-display-names.corrupt-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{broken"
    assert store.get_all({"model-good"}) == {"model-good": "恢复后名称"}


def test_display_name_store_failed_atomic_replace_keeps_existing_record(tmp_path, monkeypatch):
    store = ModelDisplayNameStore(tmp_path)
    registered = {"model-existing", "model-new"}
    store.set("model-existing", "原名称", registered_model_ids=registered)
    original = store.path.read_bytes()

    def fail_replace(*_args):
        raise OSError("simulated failure")

    monkeypatch.setattr("onegpu_posttrain.ui.model_display_names.os.replace", fail_replace)
    with pytest.raises(DisplayNameStoreError, match="原子更新"):
        store.set("model-new", "新名称", registered_model_ids=registered)

    assert store.path.read_bytes() == original
