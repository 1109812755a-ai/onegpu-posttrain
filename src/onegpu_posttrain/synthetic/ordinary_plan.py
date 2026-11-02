"""Content-addressed preregistration for non-targeted SFT generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..data.policy import digest_json


class OrdinarySFTPlan(BaseModel):
    """A normal SFT generation plan with no diagnostic/holdout evidence fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["ordinary_sft"] = "ordinary_sft"
    source_dataset_id: str = Field(min_length=1)
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_profile_id: str = Field(min_length=1)
    provider_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_id: str = Field(min_length=1)
    prompt_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_protocol_id: str = Field(min_length=1)
    output_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: str = Field(min_length=1)
    policy_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_count: int = Field(ge=1, le=10000)
    request_limit: int = Field(ge=1, le=30000)
    max_output_tokens: int = Field(default=1536, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0)

    @property
    def config_sha256(self) -> str:
        return digest_json(self.model_dump(mode="json"))

    @property
    def plan_id(self) -> str:
        return f"plan-sft-ordinary-{self.config_sha256[:20]}"
