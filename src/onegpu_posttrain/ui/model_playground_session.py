"""Ephemeral bounded chat sessions over the verified local model catalog."""

from __future__ import annotations

import gc
import math
import threading
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .gpu_lease import GpuLeaseHandle, WorkspaceGpuLease

MAX_PROMPT_CHARS = 16_000
MAX_PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 1024
MAX_HISTORY_TURNS = 8
DEFAULT_OPTIONS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "enable_thinking": False,
}


class ResolvableModelCatalog(Protocol):
    def resolve(self, model_id: str) -> dict[str, Any] | None: ...


class LocalTransformersLoader:
    """Load only a catalog-resolved local path and never execute repository code."""

    def load(self, model_path: Path) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype="auto", device_map="auto",
        )
        if hasattr(model, "eval"):
            model.eval()
        return model, tokenizer


@dataclass
class _Session:
    session_id: str
    model: dict[str, Any]
    path: Path
    model_object: Any
    tokenizer: Any
    lease: GpuLeaseHandle
    owns_lease: bool = True
    turns: list[dict[str, str]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class _CancellationCriteria:
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, *_args: Any, **_kwargs: Any) -> bool:
        return self.event.is_set()


class ModelPlaygroundSessionManager:
    """Keep one verified model and its bounded conversation in memory only."""

    def __init__(self, catalog: ResolvableModelCatalog, workspace: str | Path, *, loader: Any | None = None,
                 lease_manager: WorkspaceGpuLease | None = None):
        self.catalog = catalog
        self.workspace = Path(workspace).resolve(strict=True)
        self.loader = loader or LocalTransformersLoader()
        self.lease_manager = lease_manager or WorkspaceGpuLease(self.workspace)
        self._lock = threading.RLock()
        self._session: _Session | None = None

    @property
    def active_session(self) -> dict[str, Any] | None:
        with self._lock:
            session = self._session
            if session is None:
                return None
            return {"session_id": session.session_id, "model_id": session.model["model_id"],
                    "digest": session.model["digest"], "stage": session.model["stage"],
                    "source_run_id": session.model.get("source_run_id")}

    def open(self, model_id: str, *, session_id: str | None = None,
             lease_handle: GpuLeaseHandle | None = None) -> dict[str, Any]:
        if not isinstance(model_id, str) or len(model_id) > 160:
            raise ValueError("model_id 无效")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.isalnum() or len(session_id) > 64):
            raise ValueError("session_id 无效")
        with self._lock:
            if self._session is not None:
                raise RuntimeError("已有模型体验会话；请先结束当前会话")
            model = self.catalog.resolve(model_id)
            if not model or not model.get("available"):
                raise ValueError("模型不存在或当前不可体验")
            path_value = model.get("_resolved_path")
            if not isinstance(path_value, str) or not path_value:
                raise ValueError("模型目录没有服务端验证路径")
            path = Path(path_value).resolve(strict=True)
            # Treat the catalog as the trust boundary and re-check its result did not
            # change between resolution and loading. Callers never supply a path.
            if not path.is_dir() or not model.get("digest"):
                raise ValueError("模型目录不可用")
            owns_lease = lease_handle is None
            lease = lease_handle or self.lease_manager.acquire(
                task_type="model-chat", task_id=f"chat-{uuid.uuid4().hex[:16]}"
            )
            try:
                model_object, tokenizer = self.loader.load(path)
            except Exception as exc:
                if owns_lease:
                    lease.release()
                self._clear_cuda_cache()
                raise RuntimeError(self._failure_message(exc, "模型加载失败")) from None
            except BaseException:
                if owns_lease:
                    lease.release()
                raise
            resolved_session_id = session_id or uuid.uuid4().hex
            session = _Session(resolved_session_id, dict(model), path, model_object, tokenizer, lease, owns_lease)
            self._session = session
            return {"session_id": session.session_id, "model_id": model["model_id"],
                    "digest": model["digest"], "stage": model["stage"],
                    "source_run_id": model.get("source_run_id"), "status": "ready"}

    def chat(self, session_id: str, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"消息不能为空或超过 {MAX_PROMPT_CHARS} 个字符")
        effective = self._validate_options(options if options is not None else {})
        with self._lock:
            session = self._session
            if session is None or session.session_id != session_id:
                raise ValueError("模型体验会话不存在或已结束")
            session.cancel_event.clear()
            messages = list(session.turns) + [{"role": "user", "content": prompt}]
            if len(session.turns) // 2 >= MAX_HISTORY_TURNS:
                messages = messages[-(MAX_HISTORY_TURNS * 2 + 1):]
            thinking_supported = self._supports_thinking_mode(session.tokenizer)
            if effective["enable_thinking"] and not thinking_supported:
                raise ValueError("所选模型的聊天模板不支持思考模式控制")
            try:
                encoded = self._render(
                    session.tokenizer, messages,
                    enable_thinking=effective["enable_thinking"] if thinking_supported else None,
                )
                token_count = self._token_count(encoded)
            except Exception as exc:
                session.model_object = None
                session.tokenizer = None
                self.close(session_id)
                raise RuntimeError(self._failure_message(exc, "模型输入处理失败")) from None
            if token_count > MAX_PROMPT_TOKENS:
                raise ValueError(f"当前对话约 {token_count} tokens，超过 {MAX_PROMPT_TOKENS} 上限；请缩短上下文")
            try:
                input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
                if not hasattr(input_ids, "to"):
                    import torch
                    input_ids = torch.tensor([input_ids], dtype=torch.long)
                    encoded = {"input_ids": input_ids}
                device = getattr(session.model_object, "device", None)
                if device is None and hasattr(session.model_object, "get_input_embeddings"):
                    device = session.model_object.get_input_embeddings().weight.device
                if device is not None:
                    encoded = {key: value.to(device) if hasattr(value, "to") else value
                               for key, value in (encoded.items() if isinstance(encoded, Mapping) else {"input_ids": encoded}.items())}
                    input_ids = encoded["input_ids"]
            except Exception as exc:
                session.model_object = None
                session.tokenizer = None
                self.close(session_id)
                raise RuntimeError(self._failure_message(exc, "模型输入准备失败")) from None
            generation_options = {
                "max_new_tokens": effective["max_new_tokens"],
                "do_sample": effective["temperature"] > 0,
                "repetition_penalty": effective["repetition_penalty"],
            }
            if effective["temperature"] > 0:
                generation_options.update(temperature=effective["temperature"], top_p=effective["top_p"])
                if effective["top_k"] > 0:
                    generation_options["top_k"] = effective["top_k"]
            generation_options["stopping_criteria"] = self._stopping_criteria(session.cancel_event)
            try:
                try:
                    import torch
                    context = torch.inference_mode()
                except ImportError:
                    context = nullcontext()
                with context:
                    generated = session.model_object.generate(**encoded, **generation_options)
                if session.cancel_event.is_set():
                    raise _GenerationCancelled()
                completion_ids = generated[0][token_count:]
                answer = session.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            except _GenerationCancelled:
                self.close(session_id)
                raise RuntimeError("生成已取消") from None
            except Exception as exc:  # Do not include model prompts or raw library errors in public diagnostics.
                session.model_object = None
                session.tokenizer = None
                self.close(session_id)
                raise RuntimeError(self._failure_message(exc, "模型生成失败")) from None
            if not answer:
                self.close(session_id)
                raise RuntimeError("本地模型未返回文本")
            session.turns.extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": answer}))
            if len(session.turns) > MAX_HISTORY_TURNS * 2:
                session.turns[:] = session.turns[-MAX_HISTORY_TURNS * 2:]
            response_parameters = {
                **effective,
                "do_sample": effective["temperature"] > 0,
                "sampling_parameters_active": effective["temperature"] > 0,
                "top_p_active": effective["temperature"] > 0,
                "top_k_active": effective["temperature"] > 0 and effective["top_k"] > 0,
                "thinking_mode_supported": thinking_supported,
            }
            return {"text": answer, "model_id": session.model["model_id"],
                    "digest": session.model["digest"], "stage": session.model["stage"],
                    "source_run_id": session.model.get("source_run_id"),
                    "prompt_tokens": token_count, "parameters": response_parameters,
                    "qualitative_unverified": True}

    def compare(self, model_ids: list[str], prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one prompt against 2-3 models sequentially under a single GPU lease."""
        if not isinstance(model_ids, list) or not 2 <= len(model_ids) <= 3:
            raise ValueError("对照体验需要选择 2 到 3 个模型")
        if any(not isinstance(value, str) or not value or len(value) > 160 for value in model_ids):
            raise ValueError("model_ids 无效")
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("对照模型不能重复")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"消息不能为空或超过 {MAX_PROMPT_CHARS} 个字符")
        effective = self._validate_options(options if options is not None else {})
        resolved = []
        for model_id in model_ids:
            model = self.catalog.resolve(model_id)
            if not model or not model.get("available") or not model.get("_resolved_path"):
                raise ValueError("对照列表包含不存在或当前不可体验的模型")
            resolved.append(model)

        with self._lock:
            if self._session is not None:
                raise RuntimeError("已有模型体验会话；请先结束当前会话")
            lease = self.lease_manager.acquire(task_type="model-compare", task_id=f"compare-{uuid.uuid4().hex[:16]}")
            results = []
            try:
                for model_id in model_ids:
                    session_id = uuid.uuid4().hex
                    try:
                        self.open(model_id, session_id=session_id, lease_handle=lease)
                        result = self.chat(session_id, prompt, effective)
                        results.append({"status": "completed", **result})
                    except Exception as exc:
                        result = next((row for row in resolved if row["model_id"] == model_id), {})
                        results.append({"status": "failed", "model_id": model_id,
                                        "digest": result.get("digest"), "stage": result.get("stage"),
                                        "source_run_id": result.get("source_run_id"),
                                        "detail": (str(exc)[:240] if isinstance(exc, (ValueError, RuntimeError))
                                                   else self._failure_message(exc, "模型体验失败"))})
                        break
                    finally:
                        self.close(session_id)
            finally:
                lease.release()
        status = "completed" if all(row["status"] == "completed" for row in results) else (
            "partial" if any(row["status"] == "completed" for row in results) else "failed"
        )
        return {"status": status, "results": results, "parameters": effective,
                "qualitative_unverified": True}

    def _render(self, tokenizer: Any, messages: list[dict[str, str]], *,
                enable_thinking: bool | None = None) -> Any:
        if hasattr(tokenizer, "apply_chat_template"):
            template_options = {}
            if enable_thinking is not None:
                template_options["enable_thinking"] = enable_thinking
            return tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", **template_options,
            )
        encoded = tokenizer(messages, return_tensors="pt")
        return encoded if isinstance(encoded, dict) else {"input_ids": encoded}

    @staticmethod
    def _supports_thinking_mode(tokenizer: Any) -> bool:
        """Return true only when a tokenizer template declares the Qwen-style switch."""
        template = getattr(tokenizer, "chat_template", None)
        def declares_switch(value: Any) -> bool:
            if isinstance(value, str):
                return "enable_thinking" in value
            if isinstance(value, Mapping):
                return any(declares_switch(child) for child in value.values())
            return False
        return declares_switch(template)

    @staticmethod
    def _stopping_criteria(event: threading.Event) -> Any:
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class EventStoppingCriteria(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    return event.is_set()

            return StoppingCriteriaList([EventStoppingCriteria()])
        except ImportError:
            return [_CancellationCriteria(event)]

    @staticmethod
    def _token_count(encoded: Any) -> int:
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            return int(shape[-1])
        return len(input_ids[0] if input_ids and isinstance(input_ids[0], (list, tuple)) else input_ids)

    @staticmethod
    def _validate_options(options: dict[str, Any]) -> dict[str, int | float | bool]:
        if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS):
            raise ValueError("生成参数包含不支持的字段")
        values = {**DEFAULT_OPTIONS, **options}
        max_new_tokens = values["max_new_tokens"]
        temperature, top_p = values["temperature"], values["top_p"]
        top_k, repetition_penalty = values["top_k"], values["repetition_penalty"]
        enable_thinking = values["enable_thinking"]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_new_tokens 必须在 1 到 {MAX_OUTPUT_TOKENS} 之间")
        for name, value, low, high in (("temperature", temperature, 0.0, 2.0),
                                       ("top_p", top_p, 0.01, 1.0),
                                       ("repetition_penalty", repetition_penalty, 0.5, 2.0)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必须在 {low} 到 {high} 之间")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 100:
            raise ValueError("top_k 必须是 0 到 100 之间的整数")
        if not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking 必须是布尔值")
        return {"max_new_tokens": max_new_tokens, "temperature": float(temperature),
                "top_p": float(top_p), "top_k": top_k,
                "repetition_penalty": float(repetition_penalty), "enable_thinking": enable_thinking}

    def close(self, session_id: str | None = None) -> bool:
        active = self._session
        if active is not None and (session_id is None or active.session_id == session_id):
            active.cancel_event.set()
        with self._lock:
            session = self._session
            if session is None or (session_id is not None and session.session_id != session_id):
                return False
            self._session = None
            session.model_object = None
            session.tokenizer = None
            session.turns.clear()
            gc.collect()
            self._clear_cuda_cache()
            if session.owns_lease:
                session.lease.release()
            return True

    def cancel(self, session_id: str) -> bool:
        session = self._session
        if session is None or session.session_id != session_id:
            return False
        session.cancel_event.set()
        return True

    unload = close

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _failure_message(exc: Exception, operation: str) -> str:
        if "out of memory" in str(exc).lower() or "outofmemory" in type(exc).__name__.lower():
            return "本地 GPU 显存不足；模型会话已释放，请降低输出/上下文长度或关闭其他 GPU 任务"
        return f"本地{operation}：{type(exc).__name__}；模型会话已释放"


class _GenerationCancelled(Exception):
    pass
