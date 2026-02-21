"""Knapsack-based context packing for LLM prompts.

Given a fixed token budget and a list of candidate symbols with relevance
scores, selects the highest-value subset that fits — the classic 0/1
knapsack problem applied to LLM context window management.

Background
----------
Inspired by the comment in the r/LocalLLaMA *code-chopper* thread (2025)::

    "AST parsing and full semantic analysis using dual pipelines for text
    and code, LSP integration, and knapsack optimization so you only load
    the optimal context for the agent."

The intuition: naively stuffing all retrieved symbols into a prompt creates
two failure modes:

1. **Budget overflow** — context is truncated, often discarding the most
   relevant symbols which tend to be the largest.
2. **Information flooding** — too many candidates scatter the model's
   attention (the "lost in the middle" effect; empirically confirmed by
   Liu et al. 2023).

Knapsack packing solves both: it maximises total relevance within a hard
token cap and produces a compact, high-density context.

Typical usage
-------------
::

    from code_indexer.graph.context_packer import ContextPacker
    from code_indexer.graph.models import GraphSnapshot

    snapshot: GraphSnapshot = pipeline.snapshot

    # Score symbols by structural importance (Katz) + vector similarity
    katz = snapshot.compute_katz_centrality()
    candidates = [
        (sym, katz.get(sym.id, 0.0) * 0.4 + vector_score * 0.6)
        for sym, vector_score in vector_search_results
    ]

    packer = ContextPacker()
    packed = packer.pack(candidates, token_budget=8192)
    print(packed.summary())
    # PackedContext: 14 symbols, 7841/8192 tokens (96% utilization), 6 dropped

Algorithms
----------
``greedy``
    Sort by ``score / token_cost`` (value density) descending, greedily
    pick items until the budget is exhausted.  O(n log n).  Near-optimal
    in practice for continuous relevance distributions; preferred for
    large candidate sets.

``dp``
    Exact 0/1 knapsack via dynamic programming.  Token costs are quantised
    to ``token_quantum``-token buckets (default 64) to keep the DP table
    size tractable.  O(n × ⌈budget / quantum⌉).  Optimal but O(100×)
    slower than greedy; useful when the candidate set is small (< 200
    symbols) and optimal packing matters (e.g., very tight token budgets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from code_indexer.graph.models import SymbolNode

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class PackedContext:
    """Result of a :class:`ContextPacker` run.

    Attributes
    ----------
    symbols:
        Symbols selected to fill the context window, ordered by descending
        relevance score so the most important context appears first.
    scores:
        Relevance score for each selected symbol (parallel to ``symbols``).
    total_tokens:
        Estimated total token cost of the selected set.
    budget:
        The token budget that was applied.
    utilization:
        ``total_tokens / budget`` — how full the context window is (0–1).
        Values above 0.90 indicate tight packing; below 0.50 suggests the
        candidate set was sparse relative to the budget.
    dropped:
        How many candidate symbols were excluded (didn't fit or scored
        below ``min_score``).
    strategy:
        Which algorithm was used (``"greedy"`` or ``"dp"``).
    """

    symbols: list[SymbolNode]
    scores: list[float]
    total_tokens: int
    budget: int
    utilization: float
    dropped: int
    strategy: str

    def summary(self) -> str:
        """One-line summary suitable for logging."""
        return (
            f"PackedContext: {len(self.symbols)} symbols, "
            f"{self.total_tokens}/{self.budget} tokens "
            f"({self.utilization:.0%} utilization), "
            f"{self.dropped} dropped [{self.strategy}]"
        )

    def to_context_text(self) -> str:
        """Render all selected symbols as a plain-text context block.

        Emits one line per symbol with file path, line range, kind, and
        qualified name — suitable for injecting into an LLM prompt without
        embedding full source code.
        """
        lines: list[str] = [
            f"=== Context window: {len(self.symbols)} symbols "
            f"({self.total_tokens} est. tokens) ==="
        ]
        for sym, score in zip(self.symbols, self.scores):
            lines.append(
                f"  [{sym.kind.value}] {sym.qualified_name}"
                f"  {sym.file_path}:{sym.start_line}-{sym.end_line}"
                f"  (score={score:.3f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Packer
# ---------------------------------------------------------------------------


class ContextPacker:
    """Knapsack-based context window packer for LLM prompts.

    Selects the highest-relevance subset of candidate :class:`SymbolNode`
    objects that fits within a token budget, using either a greedy heuristic
    or exact dynamic programming.

    Parameters
    ----------
    tokens_per_line:
        Average number of tokens per source line.  Used to estimate the
        token cost of each symbol from its line span without needing the
        actual source text.  Empirical defaults by language:

        * Python / TypeScript: **15** (default)
        * Java with Javadoc: **25**
        * Rust / Go: **12**
        * C / C++: **18**
    """

    def __init__(self, tokens_per_line: float = 15.0) -> None:
        self.tokens_per_line = tokens_per_line

    # ------------------------------------------------------------------
    # Token cost estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, symbol: SymbolNode) -> int:
        """Estimate the token cost of including ``symbol`` in a prompt.

        Uses ``⌈(end_line − start_line + 1) × tokens_per_line⌉``.
        Does not require access to the source file.
        """
        lines = max(1, symbol.end_line - symbol.start_line + 1)
        return max(1, math.ceil(lines * self.tokens_per_line))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack(
        self,
        candidates: list[tuple[SymbolNode, float]],
        token_budget: int = 8192,
        *,
        strategy: Literal["greedy", "dp"] = "greedy",
        token_quantum: int = 64,
        min_score: float = 0.0,
    ) -> PackedContext:
        """Select the highest-value subset of ``candidates`` that fits within
        ``token_budget``.

        Parameters
        ----------
        candidates:
            List of ``(SymbolNode, relevance_score)`` pairs.  Scores can
            be any non-negative float — higher means more relevant.  Katz
            centrality, vector similarity, or a linear combination both
            work well::

                score = 0.4 * katz[sym.id] + 0.6 * vector_similarity

        token_budget:
            Hard upper bound on estimated total tokens.  Defaults to
            ``8_192`` — half of a typical 16 K context window, leaving
            room for the system prompt, query, and model response.
        strategy:
            ``"greedy"`` (default) — O(n log n), near-optimal.
            ``"dp"`` — exact 0/1 knapsack, slower but optimal.
        token_quantum:
            DP only.  Token costs are rounded up to this bucket size
            before building the DP table, capping table dimensions at
            ``budget // quantum`` columns.  Default ``64`` keeps the table
            at ≤ 128 columns for an 8 K budget.  Reduce for tighter
            budgets (e.g., ``16`` for a 1 K budget).
        min_score:
            Candidates with ``score < min_score`` are excluded before
            packing (default ``0.0`` — include all).

        Returns
        -------
        PackedContext
        """
        # Filter by min_score and deduplicate on symbol id
        seen: set[str] = set()
        filtered: list[tuple[SymbolNode, float, int]] = []  # (sym, score, tokens)
        for sym, score in candidates:
            if score < min_score or sym.id in seen:
                continue
            seen.add(sym.id)
            cost = self.estimate_tokens(sym)
            if cost <= token_budget:
                filtered.append((sym, score, cost))

        pre_filter_dropped = len(candidates) - len(filtered)

        if not filtered:
            return PackedContext(
                symbols=[],
                scores=[],
                total_tokens=0,
                budget=token_budget,
                utilization=0.0,
                dropped=len(candidates),
                strategy=strategy,
            )

        if strategy == "dp":
            selected = self._dp_pack(filtered, token_budget, token_quantum)
        else:
            selected = self._greedy_pack(filtered, token_budget)

        selected_ids = {sym.id for sym, _, _ in selected}
        pack_dropped = sum(1 for sym, _, _ in filtered if sym.id not in selected_ids)

        # Sort selected by descending score (highest relevance first)
        selected.sort(key=lambda t: t[1], reverse=True)

        total_tokens = sum(cost for _, _, cost in selected)
        return PackedContext(
            symbols=[sym for sym, _, _ in selected],
            scores=[score for _, score, _ in selected],
            total_tokens=total_tokens,
            budget=token_budget,
            utilization=total_tokens / token_budget if token_budget > 0 else 0.0,
            dropped=pre_filter_dropped + pack_dropped,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Greedy implementation  O(n log n)
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy_pack(
        items: list[tuple[SymbolNode, float, int]],
        budget: int,
    ) -> list[tuple[SymbolNode, float, int]]:
        """Greedy value-density knapsack heuristic.

        Sorts items by ``score / token_cost`` descending and greedily
        picks until the budget is exhausted.  This is the optimal strategy
        for the *fractional* knapsack and near-optimal for the 0/1 variant
        (~95–99% of optimal for typical code relevance distributions).
        """
        items_sorted = sorted(items, key=lambda t: t[1] / t[2], reverse=True)
        selected: list[tuple[SymbolNode, float, int]] = []
        remaining = budget
        for sym, score, cost in items_sorted:
            if cost <= remaining:
                selected.append((sym, score, cost))
                remaining -= cost
        return selected

    # ------------------------------------------------------------------
    # DP (exact) implementation  O(n × capacity)
    # ------------------------------------------------------------------

    @staticmethod
    def _dp_pack(
        items: list[tuple[SymbolNode, float, int]],
        budget: int,
        quantum: int,
    ) -> list[tuple[SymbolNode, float, int]]:
        """Exact 0/1 knapsack via dynamic programming.

        Token costs are quantised upward to the nearest ``quantum`` to
        cap the DP table to ``n × ⌊budget/quantum⌋`` cells.

        Memory: O(n × capacity) floats.  With n=200, budget=8192,
        quantum=64 → 200 × 128 = 25 600 cells ≈ 200 KB — easily fits
        in RAM.

        Backtracking recovers the exact selected set from the DP table.
        """
        # Quantise each item's cost upward to nearest quantum
        q_items: list[tuple[SymbolNode, float, int, int]] = []
        for sym, score, cost in items:
            q_cost = max(1, math.ceil(cost / quantum))
            q_items.append((sym, score, cost, q_cost))

        capacity = budget // quantum
        n = len(q_items)

        # dp[i][w] = maximum score using first i items with total weight ≤ w
        NEG_INF = float("-inf")
        dp: list[list[float]] = [[NEG_INF] * (capacity + 1) for _ in range(n + 1)]
        for w in range(capacity + 1):
            dp[0][w] = 0.0

        for i in range(1, n + 1):
            sym, score, cost, q_cost = q_items[i - 1]
            for w in range(capacity + 1):
                # Option A: don't take item i
                dp[i][w] = dp[i - 1][w]
                # Option B: take item i (if it fits and previous state was reachable)
                if q_cost <= w and dp[i - 1][w - q_cost] is not NEG_INF:
                    val_with = dp[i - 1][w - q_cost] + score
                    if val_with > dp[i][w]:
                        dp[i][w] = val_with

        # Backtrace from (n, capacity) to recover the selected items
        selected: list[tuple[SymbolNode, float, int]] = []
        w = capacity
        for i in range(n, 0, -1):
            # Item i was taken if dp[i][w] != dp[i-1][w]
            if dp[i][w] != dp[i - 1][w]:
                sym, score, cost, q_cost = q_items[i - 1]
                selected.append((sym, score, cost))
                w -= q_cost

        return selected
