"""Identifier splitting utilities for term extraction.

Code identifiers are typically abbreviated or compound words written in
camelCase, PascalCase, snake_case, or SCREAMING_SNAKE_CASE.  Before
indexing or enriching these identifiers, we split them into individual
noun tokens that can be matched against natural-language queries.

Examples
--------
>>> split_identifier("getUserByEmail")
['get', 'user', 'by', 'email']

>>> split_identifier("OrderRepository")
['order', 'repository']

>>> split_identifier("MAX_RETRY_COUNT")
['max', 'retry', 'count']

>>> split_identifier("parseHTTPResponse")
['parse', 'http', 'response']

>>> extract_noun_tokens(["getUserByEmail", "OrderRepo", "req"])
['get', 'user', 'by', 'email', 'order', 'repo', 'req']
"""

from __future__ import annotations

import re

# Minimum token length to keep.  Single letters are almost never meaningful
# noun tokens (they are loop variables, type params like T, etc.)
_MIN_TOKEN_LEN = 2

# Words that carry no domain meaning and should be stripped from noun token
# lists when used for semantic matching (prepositions, conjunctions, helpers).
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "get", "set", "is", "has", "to", "from", "by", "with",
        "for", "of", "on", "in", "at", "as", "an", "the",
        "do", "be", "or", "and", "not", "new", "all", "any",
        "can", "via", "per", "vs",
    }
)


def split_identifier(name: str) -> list[str]:
    """Split a code identifier into lowercase noun tokens.

    Handles all common naming conventions used in Python, TypeScript, Java,
    Go, and Rust:

    * ``camelCase``          → ``["camel", "case"]``
    * ``PascalCase``         → ``["pascal", "case"]``
    * ``snake_case``         → ``["snake", "case"]``
    * ``SCREAMING_SNAKE``    → ``["screaming", "snake"]``
    * ``XMLParser``          → ``["xml", "parser"]``
    * ``parseHTTPResponse``  → ``["parse", "http", "response"]``

    Parameters
    ----------
    name:
        Raw identifier string.

    Returns
    -------
    list[str]
        Lowercase tokens, each at least ``_MIN_TOKEN_LEN`` characters.
        Tokens shorter than 2 characters are dropped.
    """
    if not name:
        return []

    # Step 1: insert a space before an uppercase letter that follows a
    # lowercase letter or digit  →  handles camelCase and mixedCase.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)

    # Step 2: insert a space between a run of uppercase letters and an
    # uppercase+lowercase sequence  →  handles XMLParser → XML Parser.
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)

    # Step 3: split on any non-alphanumeric character (_, -, ., ::, etc.).
    tokens = re.split(r"[^a-zA-Z0-9]+", s)

    # Step 4: lowercase and filter short/empty tokens.
    return [t.lower() for t in tokens if len(t) >= _MIN_TOKEN_LEN]


def extract_noun_tokens(
    identifiers: list[str],
    *,
    remove_stop_words: bool = False,
) -> list[str]:
    """Extract unique noun tokens from a list of raw identifiers.

    Splits each identifier with :func:`split_identifier`, deduplicates
    across all identifiers, and preserves original order.

    Parameters
    ----------
    identifiers:
        Raw identifier strings (function names, variable names, etc.).
    remove_stop_words:
        When ``True``, generic tokens like ``get``, ``set``, ``is``,
        ``to``, ``from`` are filtered out, keeping only domain-meaningful
        nouns.  Default ``False`` — callers can decide.

    Returns
    -------
    list[str]
        Ordered, deduplicated list of lowercase noun tokens.

    Examples
    --------
    >>> extract_noun_tokens(["getUserByEmail", "sendEmail"])
    ['get', 'user', 'by', 'email', 'send']

    >>> extract_noun_tokens(["getUserByEmail"], remove_stop_words=True)
    ['user', 'email']
    """
    seen: set[str] = set()
    result: list[str] = []
    for ident in identifiers:
        for token in split_identifier(ident):
            if token in seen:
                continue
            if remove_stop_words and token in _STOP_WORDS:
                continue
            seen.add(token)
            result.append(token)
    return result


def keywords_from_query(query: str) -> list[str]:
    """Extract candidate keyword tokens from a natural-language query.

    Splits on whitespace and punctuation, lowercases, and removes stop
    words.  Used by :class:`~code_indexer.graph.term_enricher.TermKnowledgeBase`
    to match the query against indexed terms.

    Parameters
    ----------
    query:
        Natural-language question or search string.

    Returns
    -------
    list[str]
        Lowercased keyword tokens with stop words removed.

    Examples
    --------
    >>> keywords_from_query("How does JWT token verification work?")
    ['jwt', 'token', 'verification', 'work']
    """
    raw_tokens = re.split(r"[^a-zA-Z0-9]+", query.lower())
    return [
        t for t in raw_tokens
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOP_WORDS
    ]
