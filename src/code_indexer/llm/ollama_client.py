"""Ollama local LLM backend.

Calls the Ollama REST API (``/api/chat``) which must be running locally.
No external SDK required — uses the standard library ``urllib`` to avoid
adding a dependency.

Usage::

    llm = OllamaLLMClient(model="llama3.2", base_url="http://localhost:11434")
    reply = llm.complete("Expand the identifier `req`")
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from code_indexer.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECS = 60


class OllamaLLMClient(BaseLLMClient):
    """LLM client backed by a local Ollama server.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.2"``, ``"qwen2.5-coder:7b"``.
    base_url:
        Base URL of the running Ollama server.  Default: ``http://localhost:11434``.
    timeout:
        HTTP timeout in seconds (default ``60``).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = _DEFAULT_TIMEOUT_SECS,
    ) -> None:
        super().__init__(model=model)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

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
        """Call the Ollama /api/chat endpoint.

        Parameters
        ----------
        prompt:
            Prompt text sent as a single ``user`` message.
        max_tokens:
            Mapped to Ollama's ``num_predict`` option.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            Model response content, stripped of whitespace.

        Raises
        ------
        RuntimeError
            On connection errors or non-200 HTTP responses.
        """
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content: str = data.get("message", {}).get("content", "")
                return content.strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed ({self._base_url}): {exc}"
            ) from exc
