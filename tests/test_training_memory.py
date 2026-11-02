import sys
from types import SimpleNamespace

from onegpu_posttrain.ui import training_memory


def test_qlora_compatibility_reports_missing_dependency_without_installing(monkeypatch):
    def version(name):
        if name == "bitsandbytes":
            raise training_memory.importlib.metadata.PackageNotFoundError(name)
        return "2.14.1+cu126"

    monkeypatch.setattr(training_memory.importlib.metadata, "version", version)
    monkeypatch.setattr(training_memory.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                                              version=SimpleNamespace(cuda="12.6")))

    result = training_memory.inspect_training_memory_mode("qlora-4bit-nf4")

    assert result["ok"] is False
    assert result["dependencies"][0]["available"] is False
    assert "不会自动安装" in result["blockers"][0]


def test_standard_lora_has_no_optional_dependency_requirement():
    assert training_memory.inspect_training_memory_mode("lora") == {
        "ok": True, "mode_id": "lora", "dependencies": [],
    }


def test_qlora_compatibility_accepts_importable_bitsandbytes_and_cuda_torch(monkeypatch):
    monkeypatch.setattr(training_memory.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(training_memory.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(training_memory.importlib, "import_module", lambda name: SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                                              version=SimpleNamespace(cuda="12.6")))

    result = training_memory.inspect_training_memory_mode("qlora-4bit-nf4")

    assert result["ok"] is True
    assert [dependency["name"] for dependency in result["dependencies"]] == ["bitsandbytes", "torch"]
