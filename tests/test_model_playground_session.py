import pytest
import torch
import threading
import time
from collections import UserDict

from onegpu_posttrain.ui.gpu_lease import WorkspaceGpuLease
from onegpu_posttrain.ui.model_playground_session import ModelPlaygroundSessionManager


class FakeCatalog:
    def __init__(self, path):
        self.path = path
        self.model = {"model_id": "model-verified", "kind": "merged", "stage": "sft",
                      "digest": "d" * 64, "source_run_id": "run-sft-one",
                      "available": True, "_resolved_path": str(path)}

    def resolve(self, model_id):
        return dict(self.model) if model_id == self.model["model_id"] else None


class FakeTokenizer:
    def __init__(self, supports_thinking=False):
        self.rendered = []
        self.render_options = []
        self.chat_template = "{% if enable_thinking %}" if supports_thinking else "simple-chat-template"

    def apply_chat_template(self, messages, **kwargs):
        self.rendered.append([dict(message) for message in messages])
        self.render_options.append(dict(kwargs))
        count = sum(len(message["content"].split()) for message in messages) + 2
        # Transformers BatchEncoding is a Mapping/UserDict, not a plain dict.
        return UserDict({"input_ids": torch.ones((1, count), dtype=torch.long),
                         "attention_mask": torch.ones((1, count), dtype=torch.long)})

    @staticmethod
    def decode(_tokens, **_kwargs):
        return "verified answer"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.parameters = []

    def eval(self):
        self.evaluated = True

    def generate(self, input_ids, **kwargs):
        self.parameters.append(kwargs)
        return torch.cat((input_ids, torch.tensor([[7, 8]], dtype=torch.long)), dim=1)


class FakeLoader:
    def __init__(self):
        self.path = None
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()

    def load(self, model_path):
        self.path = model_path
        return self.model, self.tokenizer


def test_chat_uses_only_catalog_model_and_keeps_bounded_multi_turn_context(tmp_path):
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    loader = FakeLoader()
    catalog = FakeCatalog(model_path)
    lease = WorkspaceGpuLease(tmp_path)
    manager = ModelPlaygroundSessionManager(catalog, tmp_path, loader=loader, lease_manager=lease)

    opened = manager.open("model-verified")
    result1 = manager.chat(opened["session_id"], "first question", {"max_new_tokens": 64, "temperature": 0})
    result2 = manager.chat(opened["session_id"], "second question")

    assert loader.path == model_path.resolve()
    assert opened["model_id"] == result1["model_id"] == "model-verified"
    assert result1["digest"] == "d" * 64 and result1["source_run_id"] == "run-sft-one"
    assert result1["qualitative_unverified"] is True
    assert loader.tokenizer.rendered[1][-2:] == [
        {"role": "assistant", "content": "verified answer"},
        {"role": "user", "content": "second question"},
    ]
    assert loader.model.parameters[0]["max_new_tokens"] == 64
    assert loader.model.parameters[0]["do_sample"] is False
    assert result2["parameters"]["max_new_tokens"] == 256
    assert {key: result2["parameters"][key] for key in (
        "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "enable_thinking"
    )} == {
        "max_new_tokens": 256, "temperature": 0.7, "top_p": 0.8,
        "top_k": 20, "repetition_penalty": 1.05, "enable_thinking": False,
    }
    assert result1["parameters"]["sampling_parameters_active"] is False
    assert result1["parameters"]["top_p_active"] is False
    assert result1["parameters"]["top_k_active"] is False
    assert "top_p" not in loader.model.parameters[0] and "top_k" not in loader.model.parameters[0]
    assert lease.inspect()["status"] == "busy"
    assert manager.close(opened["session_id"]) is True
    assert lease.inspect()["status"] == "available"


def test_chat_rejects_unavailable_models_invalid_options_and_oversized_context(tmp_path):
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    loader = FakeLoader()
    catalog = FakeCatalog(model_path)
    manager = ModelPlaygroundSessionManager(catalog, tmp_path, loader=loader)
    with pytest.raises(ValueError, match="不可体验"):
        manager.open("model-not-in-catalog")
    session = manager.open("model-verified")
    with pytest.raises(ValueError, match="max_new_tokens"):
        manager.chat(session["session_id"], "hello", {"max_new_tokens": 5000})
    with pytest.raises(ValueError, match="token"):
        manager.chat(session["session_id"], "w " * 4500)
    assert manager._session.turns == []
    assert manager.close(session["session_id"])


@pytest.mark.parametrize("options, message", [
    ({"top_k": -1}, "top_k"),
    ({"top_k": 101}, "top_k"),
    ({"top_k": 1.5}, "top_k"),
    ({"enable_thinking": 1}, "enable_thinking"),
    ({"temperature": True}, "temperature"),
    ({"unsupported": 1}, "不支持"),
])
def test_chat_rejects_invalid_generation_parameter_types_and_ranges(tmp_path, options, message):
    path = tmp_path / "verified-model"
    path.mkdir()
    loader = FakeLoader()
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader)
    session = manager.open("model-verified")

    with pytest.raises(ValueError, match=message):
        manager.chat(session["session_id"], "hello", options)

    assert loader.model.parameters == []
    assert manager.close(session["session_id"])


def test_generation_parameter_defaults_and_boundaries_are_normalized():
    defaults = ModelPlaygroundSessionManager._validate_options({})
    assert defaults == {
        "max_new_tokens": 256, "temperature": 0.7, "top_p": 0.8,
        "top_k": 20, "repetition_penalty": 1.05, "enable_thinking": False,
    }
    assert ModelPlaygroundSessionManager._validate_options({"top_k": 0, "temperature": 0})["top_k"] == 0
    assert ModelPlaygroundSessionManager._validate_options({"max_new_tokens": 1024, "top_k": 100})["top_k"] == 100


def test_sampling_passes_top_k_and_top_k_zero_disables_filter(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()
    loader = FakeLoader()
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader)
    session = manager.open("model-verified")

    result = manager.chat(session["session_id"], "sample", {"top_k": 0})
    assert loader.model.parameters[-1]["do_sample"] is True
    assert loader.model.parameters[-1]["top_p"] == 0.8
    assert "top_k" not in loader.model.parameters[-1]
    assert result["parameters"]["top_k_active"] is False
    manager.close(session["session_id"])


def test_thinking_mode_is_forwarded_only_for_templates_that_declare_support(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()
    loader = FakeLoader()
    loader.tokenizer = FakeTokenizer(supports_thinking=True)
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader)
    session = manager.open("model-verified")

    result = manager.chat(session["session_id"], "think", {"enable_thinking": True})
    assert loader.tokenizer.render_options[-1]["enable_thinking"] is True
    assert result["parameters"]["thinking_mode_supported"] is True
    assert result["parameters"]["enable_thinking"] is True
    manager.close(session["session_id"])


def test_enabled_thinking_mode_rejects_unsupported_template_before_generation(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()
    loader = FakeLoader()
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader)
    session = manager.open("model-verified")

    with pytest.raises(ValueError, match="不支持思考模式"):
        manager.chat(session["session_id"], "think", {"enable_thinking": True})

    assert loader.model.parameters == []
    assert manager.active_session is not None
    assert manager.close(session["session_id"])


def test_loader_failure_releases_gpu_lease_and_does_not_create_session(tmp_path):
    class BrokenLoader:
        @staticmethod
        def load(_path):
            raise OSError("private loader diagnostic")

    path = tmp_path / "verified-model"
    path.mkdir()
    lease = WorkspaceGpuLease(tmp_path)
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=BrokenLoader(), lease_manager=lease)

    with pytest.raises(RuntimeError, match="本地模型加载失败：OSError"):
        manager.open("model-verified")
    assert manager.active_session is None
    assert lease.inspect()["status"] == "available"


def test_session_rejects_when_training_or_evaluation_holds_workspace_lease(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()
    loader = FakeLoader()
    lease = WorkspaceGpuLease(tmp_path)
    training_lease = lease.acquire(task_type="training", task_id="run-active", run_id="run-active")
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader, lease_manager=lease)

    with pytest.raises(RuntimeError, match="GPU"):
        manager.open("model-verified")

    assert loader.path is None
    training_lease.release()


def test_cancel_interrupts_generation_and_unloads_model_and_lease(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()
    entered = threading.Event()

    class SlowModel(FakeModel):
        def generate(self, input_ids, stopping_criteria, **_kwargs):
            entered.set()
            while not stopping_criteria[0](input_ids, None):
                time.sleep(0.005)
            return input_ids

    loader = FakeLoader()
    loader.model = SlowModel()
    lease = WorkspaceGpuLease(tmp_path)
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader, lease_manager=lease)
    session = manager.open("model-verified")
    failure = []

    def generate():
        try:
            manager.chat(session["session_id"], "cancel me")
        except RuntimeError as exc:
            failure.append(str(exc))

    worker = threading.Thread(target=generate)
    worker.start()
    assert entered.wait(2)
    assert manager.cancel(session["session_id"]) is True
    worker.join(3)

    assert not worker.is_alive()
    assert failure == ["生成已取消"]
    assert manager.active_session is None
    assert lease.inspect()["status"] == "available"


def test_generation_exception_is_sanitized_and_releases_loaded_resources(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()

    class BrokenModel(FakeModel):
        @staticmethod
        def generate(**_kwargs):
            raise RuntimeError("CUDA out of memory at C:\\private\\model\\secret")

    loader = FakeLoader()
    loader.model = BrokenModel()
    lease = WorkspaceGpuLease(tmp_path)
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader, lease_manager=lease)
    session = manager.open("model-verified")

    with pytest.raises(RuntimeError, match="本地 GPU 显存不足") as error:
        manager.chat(session["session_id"], "private prompt")

    assert "secret" not in str(error.value) and "private prompt" not in str(error.value)
    assert manager.active_session is None
    assert lease.inspect()["status"] == "available"


def test_comparison_loads_models_sequentially_holds_one_lease_and_keeps_partial_results(tmp_path):
    paths = [tmp_path / "model-a", tmp_path / "model-b", tmp_path / "model-c"]
    for path in paths:
        path.mkdir()

    class MultiCatalog:
        def resolve(self, model_id):
            index = {"model-a": 0, "model-b": 1, "model-c": 2}.get(model_id)
            if index is None:
                return None
            return {"model_id": model_id, "kind": "merged", "stage": "sft", "digest": str(index) * 64,
                    "source_run_id": f"run-{model_id}", "available": True, "_resolved_path": str(paths[index])}

    class ComparisonLoader(FakeLoader):
        def __init__(self, lease):
            super().__init__()
            self.lease = lease
            self.loaded = []

        def load(self, model_path):
            assert self.lease.inspect()["status"] == "busy"
            self.loaded.append(model_path.name)
            if model_path.name == "model-b":
                raise OSError("private model load detail")
            return super().load(model_path)

    lease = WorkspaceGpuLease(tmp_path)
    loader = ComparisonLoader(lease)
    manager = ModelPlaygroundSessionManager(MultiCatalog(), tmp_path, loader=loader, lease_manager=lease)
    result = manager.compare(["model-a", "model-b", "model-c"], "same prompt", {"temperature": 0})

    assert result["status"] == "partial"
    assert [row["status"] for row in result["results"]] == ["completed", "failed"]
    assert result["results"][0]["text"] == "verified answer"
    assert result["results"][0]["parameters"]["top_k"] == 20
    assert result["results"][0]["parameters"]["thinking_mode_supported"] is False
    assert "parameters" not in result["results"][1]
    assert "private model load detail" not in result["results"][1]["detail"]
    assert loader.loaded == ["model-a", "model-b"]  # C is not loaded after B fails.
    assert loader.tokenizer.rendered == [[{"role": "user", "content": "same prompt"}]]
    assert manager.active_session is None
    assert lease.inspect()["status"] == "available"


def test_comparison_rejects_duplicate_or_more_than_three_models_before_loading(tmp_path):
    path = tmp_path / "verified-model"
    path.mkdir()
    loader = FakeLoader()
    manager = ModelPlaygroundSessionManager(FakeCatalog(path), tmp_path, loader=loader)
    with pytest.raises(ValueError, match="不能重复"):
        manager.compare(["model-verified", "model-verified"], "hello")
    with pytest.raises(ValueError, match="2 到 3"):
        manager.compare(["model-verified"], "hello")
    with pytest.raises(ValueError, match="2 到 3"):
        manager.compare(["a", "b", "c", "d"], "hello")
    assert loader.path is None
