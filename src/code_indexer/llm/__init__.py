"""LLM completion clients used by TermEnricher and ConcernClusterer.

The interface mirrors the embedding layer: a thin abstract base
(``BaseLLMClient``) with concrete backends for OpenAI and Ollama.

Usage::

    from code_indexer.llm.openai_client import OpenAILLMClient
    llm = OpenAILLMClient(model="gpt-4o-mini", api_key="sk-...")
    reply = llm.complete("What does `req` stand for?")

Factory helper::

    from code_indexer.llm import make_llm_client
    from code_indexer.core.config import get_settings

    settings = get_settings()
    llm = make_llm_client(settings.llm)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_indexer.core.config import LLMSettings
    from code_indexer.llm.base import BaseLLMClient


def make_llm_client(settings: "LLMSettings") -> "BaseLLMClient":
    """Instantiate the configured LLM client from settings.

    Parameters
    ----------
    settings:
        An :class:`~code_indexer.core.config.LLMSettings` instance.

    Returns
    -------
    BaseLLMClient
        Concrete client for the configured provider.

    Raises
    ------
    ValueError
        For unknown provider names.
    """
    if settings.provider == "openai":
        from code_indexer.llm.openai_client import OpenAILLMClient  # noqa: PLC0415

        return OpenAILLMClient(
            model=settings.model,
            api_key=settings.api_key,
        )
    elif settings.provider == "ollama":
        from code_indexer.llm.ollama_client import OllamaLLMClient  # noqa: PLC0415

        return OllamaLLMClient(
            model=settings.model,
            base_url=settings.base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.provider!r}")


def make_cluster_llm_client(settings: "LLMSettings") -> "BaseLLMClient":
    """Like :func:`make_llm_client` but uses ``cluster_model`` when configured.

    The concern-clustering step benefits from a more capable model.
    Falls back to the primary model when ``cluster_model`` is empty.
    """
    cluster_model = settings.cluster_model or settings.model
    if settings.provider == "openai":
        from code_indexer.llm.openai_client import OpenAILLMClient  # noqa: PLC0415

        return OpenAILLMClient(model=cluster_model, api_key=settings.api_key)
    elif settings.provider == "ollama":
        from code_indexer.llm.ollama_client import OllamaLLMClient  # noqa: PLC0415

        return OllamaLLMClient(model=cluster_model, base_url=settings.base_url)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.provider!r}")


__all__ = ["make_llm_client", "make_cluster_llm_client"]
