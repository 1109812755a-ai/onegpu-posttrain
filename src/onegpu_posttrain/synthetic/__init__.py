"""Optional synthetic-data adapters. Importing this package performs no I/O."""

from .providers import (
    FakeProvider,
    GenerationRequest,
    GenerationResponse,
    OpenAICompatibleProvider,
    OutputProtocol,
    PromptTemplate,
    ProviderProfile,
)

__all__ = [
    "FakeProvider",
    "GenerationRequest",
    "GenerationResponse",
    "OpenAICompatibleProvider",
    "OutputProtocol",
    "PromptTemplate",
    "ProviderProfile",
]
