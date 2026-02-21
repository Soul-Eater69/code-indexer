"""Abstract base class for LLM completion backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract interface for a text-completion LLM.

    All backends expose a single ``complete`` method that takes a prompt
    string and returns the model's response as a plain string.  Callers
    are responsible for prompt construction; this layer only handles
    the API call and retry logic.

    Parameters
    ----------
    model:
        Model identifier (backend-specific).
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Run a single completion request.

        Parameters
        ----------
        prompt:
            The full prompt text.
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.  ``0.0`` → deterministic (greedy decoding).

        Returns
        -------
        str
            The model's response text, stripped of leading/trailing whitespace.

        Raises
        ------
        LLMProviderError:
            On non-retryable API errors.
        LLMRateLimitError:
            On rate-limit (HTTP 429) errors.
        """

    @property
    def name(self) -> str:
        """Human-readable identifier for logging."""
        return f"{type(self).__name__}({self.model})"
