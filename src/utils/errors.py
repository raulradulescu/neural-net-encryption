"""Shared error types and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar


class ExperimentError(RuntimeError):
    """Base class for repository-specific runtime errors."""


class ConfigError(ExperimentError):
    """Raised when config loading or merging fails."""


class ArtifactError(ExperimentError):
    """Raised when artifact serialization or filesystem writes fail."""


class ValidationError(ExperimentError):
    """Raised when input data fails a required invariant."""


TException = TypeVar("TException", bound=Exception)


def require(condition: bool, message: str, exc_type: type[TException] = ValidationError) -> None:
    """Raise ``exc_type`` if ``condition`` is false."""

    if not condition:
        raise exc_type(message)


@dataclass(frozen=True)
class ErrorContext:
    """Lightweight context container for structured error messages."""

    message: str
    hint: str | None = None

    def render(self) -> str:
        if self.hint:
            return f"{self.message}: {self.hint}"
        return self.message
