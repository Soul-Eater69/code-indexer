"""OpenAI chat-completion backend.

Uses the ``openai`` Python SDK (v1+) to call the Chat Completions API.
Rate-limit errors are retried with exponential backoff via ``tenacity``.

Recommended models
------------------
* ``gpt-4o-mini``  — default; very cheap, fast, good for bulk enrichment tasks.
* ``gpt-4o``       — use for concern clustering (``cluster_model`` setting).
* ``gpt-4.1``      — highest quality; use for maximum accuracy.
"""

from __future__ import annotations

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from code_indexer.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAILLMClient(BaseLLMClient):
    """LLM client backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    model:
        OpenAI model name (default ``gpt-4o-mini``).
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when empty.
    max_retries:
        Maximum retry attempts on transient errors (default ``4``).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        max_retries: int = 4,
    ) -> None:
        super().__init__(model=model)
        self._api_key = api_key
        self._max_retries = max_retries
        self._client: object | None = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Call the OpenAI Chat Completions API.

        Parameters
        ----------
        prompt:
            Full prompt text.  Sent as a single ``user`` message.
        max_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            Model response content, stripped of whitespace.
        """
        client = self._get_client()
        return self._call_api(client, prompt, max_tokens=max_tokens, temperature=temperature)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai  # noqa: PLC0415

                kwargs: dict = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = openai.OpenAI(**kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                ) from exc
        return self._client

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_api(
        self,
        client: object,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            response = client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content: str = response.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            exc_str = str(exc).lower()
            if "rate limit" in exc_str or "429" in exc_str:
                logger.warning("OpenAI rate limit hit, will retry: %s", exc)
            raise
