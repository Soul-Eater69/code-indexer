"""Tree-sitter based structural relationship extractor.

For each source file, this module walks the full concrete syntax tree (CST)
produced by Tree-sitter and extracts four kinds of relationships:

1. **Symbol definitions** — every function, class, method, interface, enum,
   struct, constant defined at the file level or inside a class.

2. **Import statements** — every import directive with the module string and
   the specific names being imported.  Relative imports are flagged.

3. **Call expressions** — every function/method call with the callee name and
   optional receiver object.  Call resolution (linking callee name → SymbolNode)
   happens later in :mod:`code_indexer.graph.pipeline`.

4. **Inheritance** — class/struct/trait declarations with their base
   types/interfaces.

The extractor is language-aware but uses a generic node-walking approach
to handle the many variations in tree-sitter grammar node naming.

Supported languages
-------------------
* Python
* JavaScript / TypeScript
* Rust
* Go
* Java
* C / C++ (best-effort)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from code_indexer.core.models import Language
from code_indexer.graph.models import (
    FileExtractionResult,
    RawCall,
    RawImport,
    RawInheritance,
    RawSymbol,
    SymbolKind,
)
from code_indexer.parsers.language_registry import get_tree_sitter_language

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in / noise call name sets
#
# Inspired by GitNexus's isBuiltInOrNoise pattern.  Filtering these names
# from extracted calls keeps the graph focused on user-defined symbols and
# dramatically reduces noise edges.
# ---------------------------------------------------------------------------

_PYTHON_BUILT_INS: frozenset[str] = frozenset(
    {
        # Built-in functions
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "isinstance",
        "issubclass",
        "type",
        "id",
        "hash",
        "repr",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "bytes",
        "bytearray",
        "memoryview",
        "complex",
        "frozenset",
        "object",
        "super",
        "property",
        "staticmethod",
        "classmethod",
        "abs",
        "all",
        "any",
        "bin",
        "chr",
        "dir",
        "divmod",
        "format",
        "getattr",
        "globals",
        "hasattr",
        "help",
        "hex",
        "input",
        "iter",
        "locals",
        "max",
        "min",
        "next",
        "oct",
        "open",
        "ord",
        "pow",
        "round",
        "setattr",
        "slice",
        "sorted",
        "sum",
        "vars",
        "callable",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "quit",
        "breakpoint",
        # Overly generic method names
        "append",
        "extend",
        "update",
        "get",
        "set",
        "add",
        "remove",
        "pop",
        "clear",
        "copy",
        "values",
        "keys",
        "items",
        "join",
        "split",
        "strip",
        "lstrip",
        "rstrip",
        "startswith",
        "endswith",
        "replace",
        "encode",
        "decode",
        "read",
        "write",
        "close",
        "seek",
        "tell",
        "flush",
    }
)

_JS_BUILT_INS: frozenset[str] = frozenset(
    {
        # Console / logging
        "log",
        "warn",
        "error",
        "info",
        "debug",
        "trace",
        # Array methods
        "map",
        "filter",
        "reduce",
        "forEach",
        "find",
        "findIndex",
        "some",
        "every",
        "flat",
        "flatMap",
        "includes",
        "indexOf",
        "join",
        "slice",
        "splice",
        "push",
        "pop",
        "shift",
        "unshift",
        "concat",
        "sort",
        "reverse",
        "fill",
        "copyWithin",
        # Object methods
        "keys",
        "values",
        "entries",
        "assign",
        "create",
        "freeze",
        "defineProperty",
        "getOwnPropertyNames",
        # String methods
        "toString",
        "toUpperCase",
        "toLowerCase",
        "trim",
        "trimStart",
        "trimEnd",
        "padStart",
        "padEnd",
        "repeat",
        "startsWith",
        "endsWith",
        "replace",
        "replaceAll",
        "split",
        "match",
        "search",
        "indexOf",
        "lastIndexOf",
        "charAt",
        "charCodeAt",
        "substring",
        # Promise / async
        "then",
        "catch",
        "finally",
        "resolve",
        "reject",
        "all",
        "allSettled",
        "race",
        "any",
        # React hooks
        "useState",
        "useEffect",
        "useCallback",
        "useMemo",
        "useRef",
        "useContext",
        "useReducer",
        "useLayoutEffect",
        # Global utilities
        "setTimeout",
        "setInterval",
        "clearTimeout",
        "clearInterval",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURIComponent",
        "decodeURIComponent",
        "require",
        # DOM
        "getElementById",
        "querySelector",
        "querySelectorAll",
        "addEventListener",
        "removeEventListener",
        "getAttribute",
        "setAttribute",
        "get",
        "set",
        "add",
        "delete",
        "has",
        "clear",
        "size",
    }
)

_RUST_BUILT_INS: frozenset[str] = frozenset(
    {
        "println",
        "print",
        "eprintln",
        "eprint",
        "format",
        "panic",
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "vec",
        "len",
        "push",
        "pop",
        "insert",
        "remove",
        "get",
        "iter",
        "iter_mut",
        "into_iter",
        "map",
        "filter",
        "collect",
        "unwrap",
        "expect",
        "ok",
        "err",
        "is_some",
        "is_none",
        "is_ok",
        "is_err",
        "clone",
        "into",
        "from",
        "to_string",
        "to_owned",
        "as_str",
        "as_bytes",
        "contains",
        "starts_with",
        "ends_with",
        "replace",
        "split",
        "trim",
    }
)


def _is_noise_call(name: str, language: "Language") -> bool:
    """Return True if ``name`` is a built-in or too-generic call to index."""
    if language == Language.PYTHON:
        return name in _PYTHON_BUILT_INS
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        return name in _JS_BUILT_INS
    elif language == Language.RUST:
        return name in _RUST_BUILT_INS
    # Single-letter names are always noise
    return len(name) <= 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_nodes(root: Any) -> Any:
    """Iterative DFS over tree-sitter nodes (avoids recursion limit)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _node_text(node: Any, source_bytes: bytes) -> str:
    """Return the UTF-8 text for a tree-sitter node."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _child_of_type(node: Any, *types: str) -> Any | None:
    """Return the first direct child whose type is in ``types``."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _children_of_type(node: Any, *types: str) -> list[Any]:
    """Return all direct children whose type is in ``types``."""
    return [c for c in node.children if c.type in types]


def _first_named_child(node: Any) -> Any | None:
    for child in node.children:
        if child.is_named:
            return child
    return None


def _start_line(node: Any) -> int:
    """Return 1-indexed start line."""
    return node.start_point[0] + 1


def _end_line(node: Any) -> int:
    """Return 1-indexed end line."""
    return node.end_point[0] + 1


# ---------------------------------------------------------------------------
# Language-specific extractors
# ---------------------------------------------------------------------------

# --- PYTHON -----------------------------------------------------------------


def _python_extract_imports(root: Any, src: bytes) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in _iter_nodes(root):
        if node.type == "import_statement":
            # import os  /  import os as operating_system  /  import os, sys
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(
                        RawImport(
                            module_string=_node_text(child, src),
                            imported_names=[],
                            line=_start_line(child),
                        )
                    )
                elif child.type == "aliased_import":
                    name_node = _child_of_type(child, "dotted_name")
                    alias_node = _child_of_type(child, "identifier")
                    module = _node_text(name_node, src) if name_node else ""
                    alias = _node_text(alias_node, src) if alias_node else None
                    imports.append(
                        RawImport(
                            module_string=module,
                            imported_names=[],
                            alias=alias,
                            line=_start_line(child),
                        )
                    )

        elif node.type == "import_from_statement":
            # from os.path import join, exists
            # from . import utils  (relative)
            module_node = _child_of_type(node, "dotted_name", "relative_import")
            module_str = _node_text(module_node, src) if module_node else ""
            is_relative = module_str.startswith(".")

            names: list[str] = []
            for child in node.children:
                if child.type == "dotted_name" and child is not module_node:
                    names.append(_node_text(child, src))
                elif child.type == "identifier":
                    names.append(_node_text(child, src))
                elif child.type == "wildcard_import":
                    names = ["*"]
                elif child.type == "aliased_import":
                    inner = _child_of_type(child, "dotted_name", "identifier")
                    if inner:
                        names.append(_node_text(inner, src))

            imports.append(
                RawImport(
                    module_string=module_str,
                    imported_names=names,
                    is_relative=is_relative,
                    line=_start_line(node),
                )
            )
    return imports


def _python_extract_calls(root: Any, src: bytes) -> list[RawCall]:
    calls: list[RawCall] = []
    for node in _iter_nodes(root):
        if node.type == "call":
            func_node = _child_of_type(node, "identifier", "attribute")
            if func_node is None:
                continue
            if func_node.type == "identifier":
                name = _node_text(func_node, src)
                if name in _PYTHON_BUILT_INS:
                    continue  # skip noise
                calls.append(RawCall(callee_name=name, line=_start_line(node)))
            elif func_node.type == "attribute":
                obj_node = _first_named_child(func_node)
                attr_node = _child_of_type(func_node, "identifier")
                obj_name = _node_text(obj_node, src) if obj_node else None
                attr_name = _node_text(attr_node, src) if attr_node else ""
                if attr_name in _PYTHON_BUILT_INS:
                    continue  # skip noise method calls
                calls.append(
                    RawCall(
                        callee_name=attr_name,
                        callee_object=obj_name,
                        line=_start_line(node),
                    )
                )
    return calls


def _python_extract_symbols(root: Any, src: bytes) -> list[RawSymbol]:
    symbols: list[RawSymbol] = []
    _KIND_MAP = {
        "function_definition": SymbolKind.FUNCTION,
        "async_function_def": SymbolKind.FUNCTION,
        "decorated_definition": SymbolKind.FUNCTION,
        "class_definition": SymbolKind.CLASS,
    }
    # Track current class scope for methods
    class_stack: list[str] = []

    def _walk(node: Any, depth: int) -> None:
        if node.type in ("function_definition", "async_function_def"):
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<unknown>"
            kind = SymbolKind.METHOD if class_stack else SymbolKind.FUNCTION
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=kind,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )
            # Don't descend into nested function bodies for the class_stack
        elif node.type == "class_definition":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<unknown>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.CLASS,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )
            class_stack.append(name)
            for child in node.children:
                _walk(child, depth + 1)
            class_stack.pop()
            return
        elif node.type == "decorated_definition":
            # Forward to the inner definition
            inner = _child_of_type(node, "function_definition", "class_definition")
            if inner:
                _walk(inner, depth)
            return

        for child in node.children:
            _walk(child, depth + 1)

    _walk(root, 0)
    return symbols


def _python_extract_inheritance(root: Any, src: bytes) -> list[RawInheritance]:
    result: list[RawInheritance] = []
    for node in _iter_nodes(root):
        if node.type == "class_definition":
            name_node = _child_of_type(node, "identifier")
            arg_node = _child_of_type(node, "argument_list")
            if name_node is None or arg_node is None:
                continue
            class_name = _node_text(name_node, src)
            bases = [
                _node_text(c, src)
                for c in arg_node.children
                if c.type in ("identifier", "attribute", "subscript")
            ]
            if bases:
                result.append(
                    RawInheritance(
                        class_name=class_name,
                        base_names=bases,
                        line=_start_line(node),
                    )
                )
    return result


# --- JAVASCRIPT / TYPESCRIPT ------------------------------------------------


def _js_extract_imports(root: Any, src: bytes) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in _iter_nodes(root):
        if node.type in ("import_statement", "import_declaration"):
            # import X from 'y'  /  import { A, B } from 'c'  /  import * as X from 'd'
            source_node = _child_of_type(node, "string")
            if source_node is None:
                continue
            module_str = _node_text(source_node, src).strip("'\"")
            is_relative = module_str.startswith(".")
            names: list[str] = []

            clause = _child_of_type(
                node,
                "import_clause",
                "named_imports",
                "namespace_import",
                "identifier",
            )
            if clause:
                if clause.type == "identifier":
                    names.append(_node_text(clause, src))
                elif clause.type == "namespace_import":
                    names = ["*"]
                elif clause.type in ("named_imports", "import_clause"):
                    for child in _iter_nodes(clause):
                        if child.type == "identifier" and child.parent and child.parent.type in (
                            "import_specifier",
                            "named_imports",
                        ):
                            names.append(_node_text(child, src))

            imports.append(
                RawImport(
                    module_string=module_str,
                    imported_names=names,
                    is_relative=is_relative,
                    line=_start_line(node),
                )
            )

        elif node.type == "lexical_declaration":
            # const x = require('y')
            for desc in _iter_nodes(node):
                if desc.type == "call_expression":
                    func = _child_of_type(desc, "identifier")
                    if func and _node_text(func, src) == "require":
                        args = _child_of_type(desc, "arguments")
                        if args:
                            str_node = _child_of_type(args, "string")
                            if str_node:
                                module_str = _node_text(str_node, src).strip("'\"")
                                imports.append(
                                    RawImport(
                                        module_string=module_str,
                                        is_relative=module_str.startswith("."),
                                        line=_start_line(node),
                                    )
                                )
                    break  # only one require per declaration

    return imports


def _js_extract_calls(root: Any, src: bytes) -> list[RawCall]:
    calls: list[RawCall] = []
    for node in _iter_nodes(root):
        if node.type == "call_expression":
            func = _child_of_type(node, "identifier", "member_expression")
            if func is None:
                continue
            if func.type == "identifier":
                name = _node_text(func, src)
                if name in _JS_BUILT_INS:
                    continue  # skip noise
                calls.append(RawCall(callee_name=name, line=_start_line(node)))
            elif func.type == "member_expression":
                obj = _first_named_child(func)
                prop = _child_of_type(func, "property_identifier", "identifier")
                prop_name = _node_text(prop, src) if prop else ""
                if prop_name in _JS_BUILT_INS:
                    continue  # skip noise method calls
                calls.append(
                    RawCall(
                        callee_name=prop_name,
                        callee_object=_node_text(obj, src) if obj else None,
                        line=_start_line(node),
                    )
                )
    return calls


def _js_extract_symbols(root: Any, src: bytes) -> list[RawSymbol]:
    symbols: list[RawSymbol] = []
    class_stack: list[str] = []

    def _walk(node: Any) -> None:
        if node.type in (
            "function_declaration",
            "function",
            "generator_function_declaration",
        ):
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<anonymous>"
            kind = SymbolKind.METHOD if class_stack else SymbolKind.FUNCTION
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=kind,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )
            for child in node.children:
                _walk(child)

        elif node.type == "arrow_function":
            # Try to get name from parent variable declaration
            symbols.append(
                RawSymbol(
                    name="<arrow>",
                    kind=SymbolKind.METHOD if class_stack else SymbolKind.FUNCTION,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )

        elif node.type == "method_definition":
            name_node = _child_of_type(node, "property_identifier", "identifier")
            name = _node_text(name_node, src) if name_node else "<method>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.METHOD,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )

        elif node.type == "class_declaration":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<class>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.CLASS,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )
            class_stack.append(name)
            for child in node.children:
                _walk(child)
            class_stack.pop()
            return

        elif node.type == "interface_declaration":
            name_node = _child_of_type(node, "type_identifier", "identifier")
            name = _node_text(name_node, src) if name_node else "<interface>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.INTERFACE,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )

        elif node.type == "enum_declaration":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<enum>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.ENUM,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )
        else:
            for child in node.children:
                _walk(child)
            return

        # For non-class nodes, recurse into children to find nested defs
        if node.type not in ("class_declaration",):
            for child in node.children:
                _walk(child)

    for child in root.children:
        _walk(child)
    return symbols


def _js_extract_inheritance(root: Any, src: bytes) -> list[RawInheritance]:
    result: list[RawInheritance] = []
    for node in _iter_nodes(root):
        if node.type == "class_declaration":
            name_node = _child_of_type(node, "identifier")
            heritage = _child_of_type(node, "class_heritage")
            if name_node is None or heritage is None:
                continue
            class_name = _node_text(name_node, src)
            bases = [
                _node_text(c, src)
                for c in heritage.children
                if c.type in ("identifier", "member_expression")
            ]
            if bases:
                result.append(
                    RawInheritance(
                        class_name=class_name,
                        base_names=bases,
                        line=_start_line(node),
                    )
                )
    return result


# --- RUST -------------------------------------------------------------------


def _rust_extract_imports(root: Any, src: bytes) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in _iter_nodes(root):
        if node.type == "use_declaration":
            tree_node = _child_of_type(node, "use_tree", "scoped_identifier", "identifier")
            if tree_node:
                module_str = _node_text(tree_node, src)
                imports.append(
                    RawImport(
                        module_string=module_str,
                        is_relative=module_str.startswith("self::")
                        or module_str.startswith("super::"),
                        line=_start_line(node),
                    )
                )
    return imports


def _rust_extract_calls(root: Any, src: bytes) -> list[RawCall]:
    calls: list[RawCall] = []
    for node in _iter_nodes(root):
        if node.type == "call_expression":
            func = _child_of_type(node, "identifier", "scoped_identifier", "field_expression")
            if func is None:
                continue
            if func.type == "identifier":
                name = _node_text(func, src)
                if name in _RUST_BUILT_INS:
                    continue  # skip noise
                calls.append(RawCall(callee_name=name, line=_start_line(node)))
            elif func.type == "scoped_identifier":
                # module::function
                path_parts = _node_text(func, src).rsplit("::", 1)
                callee = path_parts[-1]
                if callee in _RUST_BUILT_INS:
                    continue  # skip noise
                calls.append(
                    RawCall(
                        callee_name=callee,
                        callee_object=path_parts[0] if len(path_parts) > 1 else None,
                        line=_start_line(node),
                    )
                )
            elif func.type == "field_expression":
                field = _child_of_type(func, "field_identifier")
                obj = _first_named_child(func)
                field_name = _node_text(field, src) if field else ""
                if field_name in _RUST_BUILT_INS:
                    continue  # skip noise
                calls.append(
                    RawCall(
                        callee_name=field_name,
                        callee_object=_node_text(obj, src) if obj else None,
                        line=_start_line(node),
                    )
                )
        elif node.type == "method_call_expression":
            method = _child_of_type(node, "field_identifier")
            receiver = _first_named_child(node)
            method_name = _node_text(method, src) if method else ""
            if method_name in _RUST_BUILT_INS:
                continue  # skip noise
            calls.append(
                RawCall(
                    callee_name=method_name,
                    callee_object=_node_text(receiver, src) if receiver else None,
                    line=_start_line(node),
                )
            )
    return calls


def _rust_extract_symbols(root: Any, src: bytes) -> list[RawSymbol]:
    symbols: list[RawSymbol] = []
    impl_stack: list[str] = []

    def _walk(node: Any) -> None:
        if node.type == "function_item":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<fn>"
            kind = SymbolKind.METHOD if impl_stack else SymbolKind.FUNCTION
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=kind,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=impl_stack[-1] if impl_stack else None,
                )
            )
        elif node.type == "struct_item":
            name_node = _child_of_type(node, "type_identifier")
            name = _node_text(name_node, src) if name_node else "<struct>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.STRUCT,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )
        elif node.type == "enum_item":
            name_node = _child_of_type(node, "type_identifier")
            name = _node_text(name_node, src) if name_node else "<enum>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.ENUM,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )
        elif node.type == "trait_item":
            name_node = _child_of_type(node, "type_identifier")
            name = _node_text(name_node, src) if name_node else "<trait>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.TRAIT,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )
        elif node.type == "impl_item":
            type_node = _child_of_type(node, "type_identifier")
            impl_name = _node_text(type_node, src) if type_node else "<impl>"
            impl_stack.append(impl_name)
            for child in node.children:
                _walk(child)
            impl_stack.pop()
            return
        elif node.type == "mod_item":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<mod>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.MODULE,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )

        for child in node.children:
            _walk(child)

    for child in root.children:
        _walk(child)
    return symbols


def _rust_extract_inheritance(root: Any, src: bytes) -> list[RawInheritance]:
    # Rust doesn't have class inheritance, but traits can "extend" other traits.
    result: list[RawInheritance] = []
    for node in _iter_nodes(root):
        if node.type == "trait_item":
            name_node = _child_of_type(node, "type_identifier")
            bounds_node = _child_of_type(node, "trait_bounds")
            if name_node is None or bounds_node is None:
                continue
            bases = [
                _node_text(c, src)
                for c in bounds_node.children
                if c.type in ("type_identifier", "scoped_type_identifier")
            ]
            if bases:
                result.append(
                    RawInheritance(
                        class_name=_node_text(name_node, src),
                        base_names=bases,
                        line=_start_line(node),
                    )
                )
    return result


# --- GO ---------------------------------------------------------------------


def _go_extract_imports(root: Any, src: bytes) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in _iter_nodes(root):
        if node.type == "import_declaration":
            for spec in _iter_nodes(node):
                if spec.type == "import_spec":
                    path_node = _child_of_type(spec, "interpreted_string_literal")
                    alias_node = _child_of_type(spec, "package_identifier", "dot", "blank_identifier")
                    if path_node is None:
                        continue
                    module_str = _node_text(path_node, src).strip('"')
                    alias = _node_text(alias_node, src) if alias_node else None
                    imports.append(
                        RawImport(
                            module_string=module_str,
                            alias=alias,
                            is_relative=False,  # Go imports are always module paths
                            line=_start_line(spec),
                        )
                    )
    return imports


def _go_extract_calls(root: Any, src: bytes) -> list[RawCall]:
    calls: list[RawCall] = []
    for node in _iter_nodes(root):
        if node.type == "call_expression":
            func = _child_of_type(node, "identifier", "selector_expression")
            if func is None:
                continue
            if func.type == "identifier":
                calls.append(
                    RawCall(callee_name=_node_text(func, src), line=_start_line(node))
                )
            elif func.type == "selector_expression":
                operand = _first_named_child(func)
                field = _child_of_type(func, "field_identifier")
                calls.append(
                    RawCall(
                        callee_name=_node_text(field, src) if field else "",
                        callee_object=_node_text(operand, src) if operand else None,
                        line=_start_line(node),
                    )
                )
    return calls


def _go_extract_symbols(root: Any, src: bytes) -> list[RawSymbol]:
    symbols: list[RawSymbol] = []
    for node in _iter_nodes(root):
        if node.type == "function_declaration":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<func>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )
        elif node.type == "method_declaration":
            # func (r Receiver) Name(args) ret
            name_node = _child_of_type(node, "field_identifier")
            recv = _child_of_type(node, "parameter_list")
            receiver_type = None
            if recv:
                for c in recv.children:
                    t = _child_of_type(c, "type_identifier", "pointer_type")
                    if t:
                        receiver_type = _node_text(t, src).lstrip("*")
                        break
            name = _node_text(name_node, src) if name_node else "<method>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.METHOD,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=receiver_type,
                )
            )
        elif node.type == "type_declaration":
            for spec in _iter_nodes(node):
                if spec.type == "type_spec":
                    name_node = _child_of_type(spec, "type_identifier")
                    val = None
                    for c in spec.children:
                        if c.type in ("struct_type", "interface_type"):
                            val = c.type
                            break
                    if name_node:
                        kind = (
                            SymbolKind.STRUCT
                            if val == "struct_type"
                            else SymbolKind.INTERFACE
                            if val == "interface_type"
                            else SymbolKind.UNKNOWN
                        )
                        symbols.append(
                            RawSymbol(
                                name=_node_text(name_node, src),
                                kind=kind,
                                start_line=_start_line(spec),
                                end_line=_end_line(spec),
                            )
                        )
    return symbols


def _go_extract_inheritance(root: Any, src: bytes) -> list[RawInheritance]:
    # Go has no inheritance; interfaces are implicitly implemented
    return []


# --- JAVA -------------------------------------------------------------------


def _java_extract_imports(root: Any, src: bytes) -> list[RawImport]:
    imports: list[RawImport] = []
    for node in _iter_nodes(root):
        if node.type == "import_declaration":
            scoped = _child_of_type(node, "scoped_identifier")
            if scoped:
                module_str = _node_text(scoped, src)
                is_wildcard = module_str.endswith(".*")
                imports.append(
                    RawImport(
                        module_string=module_str,
                        imported_names=["*"] if is_wildcard else [],
                        line=_start_line(node),
                    )
                )
    return imports


def _java_extract_calls(root: Any, src: bytes) -> list[RawCall]:
    calls: list[RawCall] = []
    for node in _iter_nodes(root):
        if node.type == "method_invocation":
            name_node = _child_of_type(node, "identifier")
            obj_node = _child_of_type(node, "field_access", "identifier")
            name = _node_text(name_node, src) if name_node else ""
            obj = None
            if obj_node and obj_node is not name_node:
                obj = _node_text(obj_node, src)
            calls.append(RawCall(callee_name=name, callee_object=obj, line=_start_line(node)))
    return calls


def _java_extract_symbols(root: Any, src: bytes) -> list[RawSymbol]:
    symbols: list[RawSymbol] = []
    class_stack: list[str] = []

    def _walk(node: Any) -> None:
        if node.type == "class_declaration":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<class>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.CLASS,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )
            class_stack.append(name)
            for child in node.children:
                _walk(child)
            class_stack.pop()
            return
        elif node.type == "interface_declaration":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<interface>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.INTERFACE,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )
        elif node.type in ("method_declaration", "constructor_declaration"):
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<method>"
            kind = SymbolKind.METHOD if class_stack else SymbolKind.FUNCTION
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=kind,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    parent_name=class_stack[-1] if class_stack else None,
                )
            )
        elif node.type == "enum_declaration":
            name_node = _child_of_type(node, "identifier")
            name = _node_text(name_node, src) if name_node else "<enum>"
            symbols.append(
                RawSymbol(
                    name=name,
                    kind=SymbolKind.ENUM,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                )
            )

        for child in node.children:
            _walk(child)

    for child in root.children:
        _walk(child)
    return symbols


def _java_extract_inheritance(root: Any, src: bytes) -> list[RawInheritance]:
    result: list[RawInheritance] = []
    for node in _iter_nodes(root):
        if node.type in ("class_declaration", "interface_declaration"):
            name_node = _child_of_type(node, "identifier")
            if name_node is None:
                continue
            class_name = _node_text(name_node, src)
            bases: list[str] = []
            for child in node.children:
                if child.type in ("superclass", "super_interfaces"):
                    for c in _iter_nodes(child):
                        if c.type in ("type_identifier",):
                            bases.append(_node_text(c, src))
            if bases:
                result.append(
                    RawInheritance(
                        class_name=class_name, base_names=bases, line=_start_line(node)
                    )
                )
    return result


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

_IMPORT_EXTRACTORS = {
    Language.PYTHON: _python_extract_imports,
    Language.JAVASCRIPT: _js_extract_imports,
    Language.TYPESCRIPT: _js_extract_imports,
    Language.RUST: _rust_extract_imports,
    Language.GO: _go_extract_imports,
    Language.JAVA: _java_extract_imports,
}

_CALL_EXTRACTORS = {
    Language.PYTHON: _python_extract_calls,
    Language.JAVASCRIPT: _js_extract_calls,
    Language.TYPESCRIPT: _js_extract_calls,
    Language.RUST: _rust_extract_calls,
    Language.GO: _go_extract_calls,
    Language.JAVA: _java_extract_calls,
}

_SYMBOL_EXTRACTORS = {
    Language.PYTHON: _python_extract_symbols,
    Language.JAVASCRIPT: _js_extract_symbols,
    Language.TYPESCRIPT: _js_extract_symbols,
    Language.RUST: _rust_extract_symbols,
    Language.GO: _go_extract_symbols,
    Language.JAVA: _java_extract_symbols,
}

_INHERITANCE_EXTRACTORS = {
    Language.PYTHON: _python_extract_inheritance,
    Language.JAVASCRIPT: _js_extract_inheritance,
    Language.TYPESCRIPT: _js_extract_inheritance,
    Language.RUST: _rust_extract_inheritance,
    Language.GO: _go_extract_inheritance,
    Language.JAVA: _java_extract_inheritance,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class GraphExtractor:
    """Extract structural relationships from a source file using Tree-sitter.

    Usage::

        extractor = GraphExtractor()
        result = extractor.extract_from_source(
            file_path="src/auth.py",
            language=Language.PYTHON,
            content="def verify(token: str): ...",
        )
    """

    def extract_from_source(
        self, file_path: str, language: Language, content: str
    ) -> FileExtractionResult:
        """Parse ``content`` and extract all structural graph data.

        Returns a :class:`FileExtractionResult` with symbols, imports, calls,
        and inheritance.  Returns an empty result if the language grammar is
        not available or if parsing fails.
        """
        result = FileExtractionResult(file_path=file_path, language=language)

        lang_obj = get_tree_sitter_language(language)
        if lang_obj is None:
            logger.debug("No grammar for %s, skipping graph extraction", language.value)
            return result

        try:
            import tree_sitter  # noqa: PLC0415

            parser = tree_sitter.Parser(lang_obj)
            source_bytes = content.encode("utf-8")
            tree = parser.parse(source_bytes)
            root = tree.root_node
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tree-sitter parse error for %s: %s", file_path, exc)
            return result

        sym_fn = _SYMBOL_EXTRACTORS.get(language)
        imp_fn = _IMPORT_EXTRACTORS.get(language)
        call_fn = _CALL_EXTRACTORS.get(language)
        inh_fn = _INHERITANCE_EXTRACTORS.get(language)

        if sym_fn:
            try:
                result.symbols = sym_fn(root, source_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Symbol extraction failed for %s: %s", file_path, exc)

        if imp_fn:
            try:
                result.imports = imp_fn(root, source_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Import extraction failed for %s: %s", file_path, exc)

        if call_fn:
            try:
                result.calls = call_fn(root, source_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Call extraction failed for %s: %s", file_path, exc)

        if inh_fn:
            try:
                result.inheritance = inh_fn(root, source_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Inheritance extraction failed for %s: %s", file_path, exc)

        return result
