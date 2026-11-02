"""Conservative identity evidence for model-backed generation."""

from __future__ import annotations

from typing import Any


def provider_identity_evidence(
    *, profile_id: str | None, model: str | None, revision: str | None,
    adapter: str | None = None,
) -> dict[str, Any]:
    """Record profile declarations without claiming a verified loaded model."""
    return {
        "provider_profile_id": profile_id,
        "provider_adapter": adapter,
        "provider_model": model,
        "provider_revision": revision,
        "provider_revision_status": "available" if revision else "unavailable",
        "provider_identity_status": "profile_declared_not_independently_verified",
        "registered_model_association_status": "not_proven",
        "registered_model_id": None,
        "registered_model_sha256": None,
    }
