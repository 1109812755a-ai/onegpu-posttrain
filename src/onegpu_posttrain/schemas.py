from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CPTRecord(StrictModel):
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SFTRecord(StrictModel):
    id: str
    raw_problem: str
    messages: list[Message]
    prompt_template_id: str
    rendered_template_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_turns(self):
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("SFT record requires a user message")
        if not any(m.role == "assistant" for m in self.messages):
            raise ValueError("SFT record requires an assistant target")
        return self


class PreferenceRecord(StrictModel):
    id: str
    raw_problem: str
    prompt: list[Message]
    chosen: str
    rejected: str
    prompt_template_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pair(self):
        if not self.chosen.strip() or not self.rejected.strip():
            raise ValueError("chosen/rejected must be non-empty")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen/rejected must differ")
        return self


class GRPORecord(StrictModel):
    id: str
    raw_problem: str
    prompt: list[Message]
    answer: str
    prompt_template_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(StrictModel):
    id: str
    problem_id: str
    raw_problem: str
    prompt: str
    answer: str
    eval_prompt_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    question_type: Literal["single_choice", "multiple_choice", "true_false"] | None = None
    options: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_objective_choice(self):
        if self.question_type is None:
            return self
        if self.question_type == "true_false":
            if self.options:
                raise ValueError("true_false records must not declare options")
            if self.answer not in {"true", "false"}:
                raise ValueError("true_false answer must be true or false")
            return self
        if len(self.options) < 2 or len(self.options) > 26:
            raise ValueError("choice records require 2 to 26 options")
        labels = {chr(ord("A") + index) for index in range(len(self.options))}
        answers = self.answer.split(",")
        if not answers or any(item not in labels for item in answers) or len(answers) != len(set(answers)):
            raise ValueError("choice answer contains an invalid option label")
        if self.question_type == "single_choice" and len(answers) != 1:
            raise ValueError("single_choice answer must contain one option")
        if self.question_type == "multiple_choice" and not answers:
            raise ValueError("multiple_choice answer must not be empty")
        return self


STAGE_MODELS = {
    "cpt": CPTRecord,
    "sft": SFTRecord,
    "preference": PreferenceRecord,
    "grpo": GRPORecord,
    "eval": EvalRecord,
}
