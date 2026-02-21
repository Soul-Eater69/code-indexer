"""Query-time concern clustering for RAG context construction.

Inspired by the online retrieval-and-ranking stage of RepoLens
(Wang et al., arXiv:2509.21427).

The problem: concern scattering
--------------------------------
When the logic relevant to a bug or feature is spread across many files,
a ranked list of individual symbols hides the structure.  Symbols 1, 4,
and 6 in the list may all belong to the same cross-cutting concern
(e.g. "JWT authentication"), but a flat list doesn't make this visible.

The solution: concern clustering
---------------------------------
After vector retrieval, pass the retrieved :class:`TermChunk` objects to
:class:`ConcernClusterer`.  It uses a capable LLM to group them into 2–5
named concerns — coherent feature clusters like "Token validation logic"
or "Session invalidation on logout".

The resulting :class:`Concern` objects are then injected into the LLM
prompt as **soft guidance**: high-level pointers that tell the model where
to look, while still allowing it to explore the codebase autonomously if
the concerns turn out to be noisy.

Design principle: guide, don't restrict
-----------------------------------------
The concern block is prepended to the prompt with a clear instruction that
it represents inferred hints, not ground truth.  The LLM retains full
autonomy to search, grep, and browse — the concerns just provide orientation.

Usage::

    from code_indexer.llm import make_llm_client, make_cluster_llm_client
    from code_indexer.graph.concern_clusterer import ConcernClusterer

    llm = make_llm_client(settings.llm)
    cluster_llm = make_cluster_llm_client(settings.llm)
    clusterer = ConcernClusterer(llm=llm, cluster_llm=cluster_llm)

    term_chunks_with_scores: list[tuple[TermChunk, float]] = [...]  # from retrieval
    concerns = clusterer.cluster(term_chunks_with_scores, query="How does auth work?")
    for c in concerns:
        print(c.to_prompt_block())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from code_indexer.graph.models import SymbolNode
from code_indexer.graph.term_chunker import TermChunk

if TYPE_CHECKING:
    from code_indexer.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concern data model
# ---------------------------------------------------------------------------


@dataclass
class Concern:
    """A semantically coherent cluster of code functionalities.

    Groups multiple :class:`TermChunk` objects (and the symbols behind them)
    that together form a logical feature or cross-cutting aspect of the
    codebase relevant to the current query.

    Attributes
    ----------
    name:
        Short, descriptive name for the concern (3–6 words),
        e.g. ``"JWT token validation logic"``.
    description:
        One-sentence explanation of what this concern covers and why
        it is relevant to the current query.
    symbols:
        Deduplicated list of :class:`~code_indexer.graph.models.SymbolNode`
        objects that belong to this concern.
    term_chunks:
        The underlying :class:`TermChunk` objects that were clustered
        into this concern.
    """

    name: str
    description: str
    symbols: list[SymbolNode] = field(default_factory=list)
    term_chunks: list[TermChunk] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def to_prompt_block(self) -> str:
        """Render as a soft-guidance block for LLM system/user prompts.

        Example output::

            Concern: JWT token validation logic
              Handles parsing, signature verification, and expiry checking
              of JWT tokens in incoming API requests.
              Relevant code (3 symbols):
                • JWTHandler.verify_token  (src/auth/jwt.py:42)
                • decode_payload  (src/auth/jwt.py:89)
                • get_current_user  (src/api/deps.py:15)
        """
        lines = [
            f"Concern: {self.name}",
            f"  {self.description}",
            f"  Relevant code ({len(self.symbols)} symbols):",
        ]
        for sym in self.symbols[:8]:
            lines.append(
                f"    • {sym.qualified_name}"
                f"  ({sym.file_path}:{sym.start_line})"
            )
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Render a Mermaid flowchart of the symbols within this concern.

        Useful for visual debugging and for injecting a structural picture
        into LLM prompts alongside the text description.

        Example output::

            flowchart TD
                _title["JWT token validation logic"]:::concern
                _sym_0["verify_token [method]"]
                _title --- _sym_0
                _sym_1["decode_payload [function]"]
                _title --- _sym_1
                classDef concern fill:#4a9,stroke:#063,color:#fff
        """

        def _lbl(sym: SymbolNode) -> str:
            return f"{sym.name} [{sym.kind.value}]".replace('"', "'")

        safe_name = self.name.replace('"', "'")
        lines = [
            "flowchart TD",
            f'    _title["{safe_name}"]:::concern',
        ]
        for i, sym in enumerate(self.symbols[:10]):
            nid = f"_sym_{i}"
            lines.append(f'    {nid}["{_lbl(sym)}"]')
            lines.append(f"    _title --- {nid}")
        lines.append("    classDef concern fill:#4a9,stroke:#063,color:#fff")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConcernClusterer
# ---------------------------------------------------------------------------


class ConcernClusterer:
    """Cluster retrieved TermChunks into high-level conceptual concerns.

    At query time, takes the list of ``(TermChunk, score)`` pairs produced
    by vector retrieval and uses a capable LLM to group them into 2–5 named
    concerns.  A second, cheaper LLM call then ranks the concerns by
    relevance to the query.

    Two LLM clients are used deliberately:

    * ``cluster_llm`` — a more capable model (e.g. ``gpt-4o``) for the
      clustering step where semantic reasoning quality matters most.
    * ``llm`` — a cheap model (e.g. ``gpt-4o-mini``) for the ranking step,
      which is simpler and more mechanical.

    When only one client is provided, it handles both steps.

    Parameters
    ----------
    llm:
        LLM client for cheap operations (ranking, fallback clustering).
    cluster_llm:
        Optional stronger LLM client for the clustering step.
        Defaults to ``llm`` when ``None``.
    """

    _CLUSTER_PROMPT = """\
You are analysing a software codebase to answer this question:
"{query}"

Here are relevant code functionalities retrieved from the codebase:

{functionalities}

Group these functionalities into 2–5 high-level conceptual concerns.
Each concern should be a coherent feature or cross-cutting aspect of the codebase.

Reply with a JSON array (no markdown fences). Each element must have exactly:
  "name": short concern name (3–6 words)
  "description": one sentence describing what this concern covers
  "indices": list of 0-based indices from the functionalities above

Reply with ONLY the JSON array."""

    _RANK_PROMPT = """\
You are ranking software concerns by relevance to this question:
"{query}"

Concerns:
{concerns}

Return ONLY a JSON array of 0-based integer indices, ordered from most to least \
relevant to the question. Example: [2, 0, 1]"""

    def __init__(
        self,
        llm: "BaseLLMClient",
        *,
        cluster_llm: "BaseLLMClient | None" = None,
    ) -> None:
        self._llm = llm
        self._cluster_llm = cluster_llm or llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        term_chunks: list[tuple[TermChunk, float]],
        query: str,
        *,
        max_concerns: int = 5,
        max_input_chunks: int = 20,
    ) -> list[Concern]:
        """Cluster retrieved term chunks into concerns for the given query.

        Parameters
        ----------
        term_chunks:
            ``(TermChunk, relevance_score)`` pairs — typically from vector
            retrieval + scoring.  Only the top ``max_input_chunks`` are
            included in the clustering prompt.
        query:
            The user's natural-language question.
        max_concerns:
            Maximum number of concerns to return (default ``5``).
        max_input_chunks:
            Maximum number of term chunks to include in the clustering
            prompt (default ``20``).  Higher values improve coverage but
            increase LLM cost.

        Returns
        -------
        list[Concern]
            Concerns ranked by relevance (most relevant first).
            Empty list on LLM failure.
        """
        if not term_chunks:
            return []

        # Cap input to keep prompt size predictable
        capped = term_chunks[:max_input_chunks]

        # Build numbered functionality lines for the clustering prompt
        fn_lines = []
        for i, (chunk, _score) in enumerate(capped):
            fn_lines.append(
                f"{i}. [{chunk.symbol.kind.value}] {chunk.symbol.qualified_name}"
                f" | term: {chunk.term.expanded_name}"
                f" | {chunk.summary}"
            )

        cluster_prompt = self._CLUSTER_PROMPT.format(
            query=query,
            functionalities="\n".join(fn_lines),
        )

        # Step 1: cluster with the stronger model
        try:
            raw = self._cluster_llm.complete(cluster_prompt, max_tokens=700)
            cluster_data = self._parse_json(raw)
        except Exception:  # noqa: BLE001
            logger.warning("ConcernClusterer: clustering LLM call failed")
            return []

        if not isinstance(cluster_data, list):
            logger.warning(
                "ConcernClusterer: expected JSON array, got %s", type(cluster_data)
            )
            return []

        # Step 2: convert raw JSON to Concern objects
        chunks_list = [tc for tc, _ in capped]
        concerns = self._build_concerns(cluster_data, chunks_list, max_concerns)

        if len(concerns) <= 1:
            return concerns

        # Step 3: rank by relevance with the cheap model
        try:
            concerns = self._rank_concerns(concerns, query)
        except Exception:  # noqa: BLE001
            logger.debug(
                "ConcernClusterer: ranking step failed, using cluster order"
            )

        return concerns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_concerns(
        self,
        cluster_data: list,
        chunks_list: list[TermChunk],
        max_concerns: int,
    ) -> list[Concern]:
        concerns: list[Concern] = []
        for item in cluster_data[:max_concerns]:
            if not isinstance(item, dict):
                continue
            indices = item.get("indices", [])
            if not isinstance(indices, list):
                continue

            selected_chunks = [
                chunks_list[i]
                for i in indices
                if isinstance(i, int) and 0 <= i < len(chunks_list)
            ]
            if not selected_chunks:
                continue

            # Deduplicate symbols while preserving first-seen order
            seen_ids: set[str] = set()
            symbols: list[SymbolNode] = []
            for chunk in selected_chunks:
                if chunk.symbol.id not in seen_ids:
                    seen_ids.add(chunk.symbol.id)
                    symbols.append(chunk.symbol)

            concerns.append(
                Concern(
                    name=str(item.get("name", f"Concern {len(concerns) + 1}")),
                    description=str(item.get("description", "")),
                    symbols=symbols,
                    term_chunks=selected_chunks,
                )
            )
        return concerns

    def _rank_concerns(
        self, concerns: list[Concern], query: str
    ) -> list[Concern]:
        """Reorder concerns by relevance to query using the cheap LLM."""
        concern_lines = "\n".join(
            f"{i}. {c.name}: {c.description}" for i, c in enumerate(concerns)
        )
        rank_prompt = self._RANK_PROMPT.format(
            query=query, concerns=concern_lines
        )
        raw = self._llm.complete(rank_prompt, max_tokens=60)
        order = self._parse_json(raw)

        if not isinstance(order, list):
            return concerns
        if not all(isinstance(x, int) for x in order):
            return concerns

        reordered: list[Concern] = []
        seen: set[int] = set()
        for i in order:
            if 0 <= i < len(concerns) and i not in seen:
                reordered.append(concerns[i])
                seen.add(i)
        # Append any concerns not mentioned in the ranked order
        for i, c in enumerate(concerns):
            if i not in seen:
                reordered.append(c)
        return reordered

    @staticmethod
    def _parse_json(text: str) -> object:
        """Parse JSON from LLM output, tolerating markdown code fences."""
        text = text.strip()
        # Strip ``` or ```json fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (the fence opener) and last line (the closer)
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            text = "\n".join(inner_lines).strip()
        return json.loads(text)
