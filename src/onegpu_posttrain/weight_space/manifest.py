"""Versioned, immutable identities for an I Ching weight-space experiment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "05f056712f8c582c8708395156518d963b46fb29"
MODEL_ID = "model-89d12ba9eb8fb14d6eb6"
MODEL_CHECKPOINT_SHA256 = "89d12ba9eb8fb14d6eb6b8b2ea85688cd254d98f64f2bfeee85db0f3d741a4be"
MODEL_PROCESSOR_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
DOCKER_IMAGE = (
    "hiyouga/llamafactory@sha256:"
    "2167de797ca005b1c9b013820e0264db17b799945588cdfe11e9d78cc42576a4"
)
PRACTICE_DATASET_ID = "ds-eval-upload-14aee30e5c7ec6ec0959"
PRACTICE_DATASET_SHA256 = "361c56f60059392c82e092762121c6ef50d53496dc0f6902cdd330f304b2c0be"
FINAL_DATASET_ID = "ds-eval-upload-54ca42ca021d295d8390"
FINAL_DATASET_SHA256 = "19f02eab2e2d3894269eda8f41115c22c2f203cfe3c3cbb3604ab19488004ac2"
SHA256 = r"^[0-9a-f]{64}$"


class IdentityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class TargetModel(IdentityModel):
    model_id: str = Field(min_length=1)
    repository: Literal["Qwen/Qwen3.5-0.8B"]
    revision: Literal["05f056712f8c582c8708395156518d963b46fb29"]
    checkpoint_sha256: str = Field(pattern=SHA256)
    processor_sha256: str = Field(pattern=SHA256)


class Runtime(IdentityModel):
    image: Literal[
        "hiyouga/llamafactory@sha256:2167de797ca005b1c9b013820e0264db17b799945588cdfe11e9d78cc42576a4"
    ]
    torch_version: Literal["2.6.0+cu124"]
    transformers_version: Literal["5.8.0"]
    model_loader: Literal["AutoProcessor+AutoModelForMultimodalLM"]


class Dataset(IdentityModel):
    dataset_id: str = Field(min_length=1)
    role: Literal["practice", "final_holdout"]
    canonical_sha256: str = Field(pattern=SHA256)
    record_count: int = Field(gt=0)


class Inference(IdentityModel):
    prompt_template: Literal["qwen3.5-processor-chat-template"]
    scoring_protocol: Literal["choice-accuracy-v1"]
    max_new_tokens: Literal[16]
    do_sample: Literal[False]
    skip_special_tokens: Literal[True]


class Baseline(IdentityModel):
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    input_sha256: str = Field(pattern=SHA256)
    correct: int = Field(ge=0)
    denominator: int = Field(gt=0)
    parse_failures: int = Field(ge=0)
    evidence: Literal["observed_container_run", "registered_evaluation_run"]
    evaluation_run_id: str | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> Baseline:
        if self.correct > self.denominator or self.parse_failures > self.denominator:
            raise ValueError("baseline counts exceed denominator")
        if self.evidence == "registered_evaluation_run" and not self.evaluation_run_id:
            raise ValueError("registered baseline needs evaluation run ID")
        return self


class Iteration(IdentityModel):
    index: int = Field(ge=1)
    parent_checkpoint_sha256: str = Field(pattern=SHA256)
    candidate_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256)
    practice_dataset_id: str = Field(min_length=1)
    editor_model: Literal["qwen3.8-max"]
    status: Literal["planned", "running", "completed", "failed", "discarded"]


class ExperimentManifest(IdentityModel):
    schema_version: Literal["1.0"]
    experiment_id: str = Field(pattern=r"^exp-weight-space-[a-z0-9-]{8,64}$")
    target: TargetModel
    runtime: Runtime
    practice: Dataset
    final_holdout: Dataset
    inference: Inference
    baseline: Baseline
    editor_model: Literal["qwen3.8-max"]
    iterations: list[Iteration] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> ExperimentManifest:
        if self.target.model_id != MODEL_ID:
            raise ValueError("unexpected target model ID")
        if (self.target.checkpoint_sha256 != MODEL_CHECKPOINT_SHA256
                or self.target.processor_sha256 != MODEL_PROCESSOR_SHA256):
            raise ValueError("target checkpoint or processor digest does not match pin")
        if self.practice.role != "practice" or self.final_holdout.role != "final_holdout":
            raise ValueError("practice and final roles are fixed")
        if self.practice.dataset_id != PRACTICE_DATASET_ID or self.final_holdout.dataset_id != FINAL_DATASET_ID:
            raise ValueError("unexpected dataset ID")
        if (self.practice.canonical_sha256 != PRACTICE_DATASET_SHA256
                or self.final_holdout.canonical_sha256 != FINAL_DATASET_SHA256):
            raise ValueError("dataset digest does not match pin")
        if (self.practice.dataset_id == self.final_holdout.dataset_id
                or self.practice.canonical_sha256 == self.final_holdout.canonical_sha256):
            raise ValueError("practice and final datasets must be disjoint")
        if (self.baseline.model_id != self.target.model_id
                or self.baseline.dataset_id != self.final_holdout.dataset_id
                or self.baseline.input_sha256 != self.final_holdout.canonical_sha256
                or self.baseline.denominator != self.final_holdout.record_count):
            raise ValueError("baseline identity does not match target and final holdout")
        if len({item.index for item in self.iterations}) != len(self.iterations):
            raise ValueError("iteration indexes must be unique")
        if any(item.practice_dataset_id != self.practice.dataset_id for item in self.iterations):
            raise ValueError("iteration must use the pinned practice dataset")
        return self
