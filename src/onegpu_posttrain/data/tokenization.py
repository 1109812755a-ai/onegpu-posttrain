from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class TokenizerBackend(Protocol):
    name: str

    def count_text(self, text: str) -> int: ...
    def render_messages(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> str: ...
    def count_messages(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> int: ...
    def chunk_text(self, text: str, max_tokens: int) -> list[str]: ...


@dataclass
class WhitespaceTokenizer:
    """Deterministic test/offline tokenizer. Never use it for official length gates."""

    name: str = "whitespace-test-only"

    def count_text(self, text: str) -> int:
        return len(text.split())

    def render_messages(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> str:
        rendered = "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\n<assistant>\n"
        return rendered

    def count_messages(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> int:
        return self.count_text(self.render_messages(messages, add_generation_prompt=add_generation_prompt))

    def chunk_text(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens) if words[i : i + max_tokens]]


class HuggingFaceTokenizer:
    """Pinned HF tokenizer used by the official Length Eligibility Gate."""

    def __init__(self, repo: str, revision: str | None = None, *, local_files_only: bool = False):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install onegpu-posttrain[data] to use HuggingFaceTokenizer") from exc
        self.repo = repo
        self.revision = revision
        self.name = f"{repo}@{revision or 'main'}"
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=False,
                                                       local_files_only=local_files_only)

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render_messages(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> str:
        payload = [dict(m) for m in messages]
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            return self.tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Defensive fallback for tokenizers without a chat template.
        text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in payload)
        return text + ("\nASSISTANT:" if add_generation_prompt else "")

    def count_messages(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> int:
        rendered = self.render_messages(messages, add_generation_prompt=add_generation_prompt)
        return self.count_text(rendered)

    def chunk_text(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: list[str] = []
        for i in range(0, len(ids), max_tokens):
            piece = ids[i : i + max_tokens]
            if piece:
                chunks.append(self.tokenizer.decode(piece, skip_special_tokens=False))
        return chunks
