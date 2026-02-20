"""Tests for the InMemoryGraphStore."""

import pytest

from code_indexer.core.models import Language
from code_indexer.graph.in_memory_graph import InMemoryGraphStore
from code_indexer.graph.models import (
    FileNode,
    GraphEdge,
    GraphSnapshot,
    RelType,
    SymbolKind,
    SymbolNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryGraphStore:
    """Fresh InMemoryGraphStore (no NetworkX required)."""
    return InMemoryGraphStore(use_networkx=False)


def _make_file(path: str, language: Language = Language.PYTHON) -> FileNode:
    return FileNode.from_path(path, language, sha256="deadbeef" * 4)


def _make_symbol(
    name: str,
    file_id: str,
    file_path: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
    parent_name: str | None = None,
) -> SymbolNode:
    qname = f"{file_path}::{(parent_name + '.' if parent_name else '')}{name}"
    return SymbolNode(
        id=SymbolNode.make_id(file_id, qname),
        name=name,
        qualified_name=qname,
        kind=kind,
        file_id=file_id,
        file_path=file_path,
        start_line=1,
        end_line=10,
        parent_name=parent_name,
    )


def _make_snapshot() -> GraphSnapshot:
    """Build a small but realistic graph snapshot.

    Files:
        auth.py → defines: verify_token, User (class), User.check_password
        utils.py → defines: hash_password
    Edges:
        auth.py IMPORTS utils.py
        verify_token CALLS hash_password
        User CONTAINS check_password
    """
    snap = GraphSnapshot()

    auth_file = _make_file("auth.py")
    utils_file = _make_file("utils.py")
    snap.file_nodes[auth_file.id] = auth_file
    snap.file_nodes[utils_file.id] = utils_file

    verify = _make_symbol("verify_token", auth_file.id, "auth.py")
    user_cls = _make_symbol("User", auth_file.id, "auth.py", kind=SymbolKind.CLASS)
    check_pw = _make_symbol(
        "check_password", auth_file.id, "auth.py", kind=SymbolKind.METHOD, parent_name="User"
    )
    hash_pw = _make_symbol("hash_password", utils_file.id, "utils.py")
    snap.symbol_nodes[verify.id] = verify
    snap.symbol_nodes[user_cls.id] = user_cls
    snap.symbol_nodes[check_pw.id] = check_pw
    snap.symbol_nodes[hash_pw.id] = hash_pw

    # auth.py IMPORTS utils.py
    snap.edges.append(
        GraphEdge.create(RelType.IMPORTS, auth_file.id, utils_file.id, module_string="utils")
    )
    # File DEFINES symbols
    snap.edges.append(GraphEdge.create(RelType.DEFINES, auth_file.id, verify.id))
    snap.edges.append(GraphEdge.create(RelType.DEFINES, auth_file.id, user_cls.id))
    snap.edges.append(GraphEdge.create(RelType.DEFINES, auth_file.id, check_pw.id))
    snap.edges.append(GraphEdge.create(RelType.DEFINES, utils_file.id, hash_pw.id))
    # User CONTAINS check_password
    snap.edges.append(GraphEdge.create(RelType.CONTAINS, user_cls.id, check_pw.id))
    # verify_token CALLS hash_password
    snap.edges.append(
        GraphEdge.create(RelType.CALLS, verify.id, hash_pw.id, line=5)
    )

    return snap


# ---------------------------------------------------------------------------
# Upsert & basic lookup
# ---------------------------------------------------------------------------


def test_upsert_then_find_symbol(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    results = store.find_symbols_by_name("verify_token")
    assert len(results) == 1
    assert results[0].name == "verify_token"


def test_upsert_idempotent(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    store.upsert(snap)  # second upsert must not duplicate
    results = store.find_symbols_by_name("hash_password")
    assert len(results) == 1


def test_find_symbols_in_file(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    auth_file = next(fn for fn in snap.file_nodes.values() if fn.path == "auth.py")
    syms = store.find_symbols_in_file(auth_file.id)
    names = {s.name for s in syms}
    assert {"verify_token", "User", "check_password"} == names


def test_get_symbol_by_id(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    hash_pw = next(s for s in snap.symbol_nodes.values() if s.name == "hash_password")
    fetched = store.get_symbol_by_id(hash_pw.id)
    assert fetched is not None
    assert fetched.name == "hash_password"


def test_get_symbol_by_id_missing(store: InMemoryGraphStore) -> None:
    store.upsert(_make_snapshot())
    assert store.get_symbol_by_id("nonexistent") is None


# ---------------------------------------------------------------------------
# Call graph traversal
# ---------------------------------------------------------------------------


def test_callers(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    hash_pw = next(s for s in snap.symbol_nodes.values() if s.name == "hash_password")
    callers = store.get_callers(hash_pw.id)
    assert len(callers) == 1
    assert callers[0].name == "verify_token"


def test_callees(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    verify = next(s for s in snap.symbol_nodes.values() if s.name == "verify_token")
    callees = store.get_callees(verify.id)
    assert len(callees) == 1
    assert callees[0].name == "hash_password"


def test_no_callers(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    verify = next(s for s in snap.symbol_nodes.values() if s.name == "verify_token")
    callers = store.get_callers(verify.id)
    assert callers == []


# ---------------------------------------------------------------------------
# Call path (BFS)
# ---------------------------------------------------------------------------


def test_call_path_direct(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    verify = next(s for s in snap.symbol_nodes.values() if s.name == "verify_token")
    hash_pw = next(s for s in snap.symbol_nodes.values() if s.name == "hash_password")
    path = store.get_call_path(verify.id, hash_pw.id)
    assert path is not None
    names = [s.name for s in path]
    assert names == ["verify_token", "hash_password"]


def test_call_path_no_path(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    hash_pw = next(s for s in snap.symbol_nodes.values() if s.name == "hash_password")
    user_cls = next(s for s in snap.symbol_nodes.values() if s.name == "User")
    path = store.get_call_path(hash_pw.id, user_cls.id)
    assert path is None


# ---------------------------------------------------------------------------
# Import graph
# ---------------------------------------------------------------------------


def test_import_graph(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    auth_file = next(fn for fn in snap.file_nodes.values() if fn.path == "auth.py")
    graph = store.get_import_graph(auth_file.id)
    assert "auth.py" in graph
    assert "utils.py" in graph.get("auth.py", [])


# ---------------------------------------------------------------------------
# Containment (class → method)
# ---------------------------------------------------------------------------


def test_class_contains_method(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    user_cls = next(s for s in snap.symbol_nodes.values() if s.name == "User")
    ctx = store.get_symbol_context(user_cls.id)
    assert ctx is not None
    method_names = [m.name for m in ctx.methods]
    assert "check_password" in method_names


# ---------------------------------------------------------------------------
# Symbol context
# ---------------------------------------------------------------------------


def test_symbol_context_full(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    hash_pw = next(s for s in snap.symbol_nodes.values() if s.name == "hash_password")
    ctx = store.get_symbol_context(hash_pw.id)
    assert ctx is not None
    assert ctx.symbol.name == "hash_password"
    assert ctx.file.path == "utils.py"
    # verify_token calls hash_password → it should appear as a caller
    caller_names = [c.name for c in ctx.callers]
    assert "verify_token" in caller_names


def test_symbol_context_file_imports(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    verify = next(s for s in snap.symbol_nodes.values() if s.name == "verify_token")
    ctx = store.get_symbol_context(verify.id)
    assert ctx is not None
    imported_paths = [f.path for f in ctx.file_imports]
    assert "utils.py" in imported_paths


# ---------------------------------------------------------------------------
# Delete and clear
# ---------------------------------------------------------------------------


def test_delete_file_removes_symbols(store: InMemoryGraphStore) -> None:
    snap = _make_snapshot()
    store.upsert(snap)
    utils_file = next(fn for fn in snap.file_nodes.values() if fn.path == "utils.py")
    removed = store.delete_file(utils_file.id)
    assert removed > 0
    assert store.find_symbols_by_name("hash_password") == []


def test_clear(store: InMemoryGraphStore) -> None:
    store.upsert(_make_snapshot())
    store.clear()
    assert store.find_symbols_by_name("verify_token") == []
    stats = store.get_stats()
    assert stats.total_files == 0
    assert stats.total_symbols == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_stats(store: InMemoryGraphStore) -> None:
    store.upsert(_make_snapshot())
    stats = store.get_stats()
    assert stats.total_files == 2
    assert stats.total_symbols == 4
    assert stats.total_edges > 0
    assert "function" in stats.symbols_by_kind or "method" in stats.symbols_by_kind
    assert "python" in stats.languages
