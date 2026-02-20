"""Tests for the Tree-sitter graph extractor."""

import pytest

from code_indexer.core.models import Language
from code_indexer.graph.extractor import GraphExtractor
from code_indexer.graph.models import SymbolKind


@pytest.fixture
def extractor() -> GraphExtractor:
    return GraphExtractor()


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------

PYTHON_SOURCE = """\
import os
import sys as system
from os.path import join, exists
from . import utils
from ..models import User

CONSTANT = 42

def top_level_func(x, y):
    return x + y

async def async_func():
    pass

class MyClass(Base, Mixin):
    def __init__(self):
        self.value = 0
        top_level_func(1, 2)

    def method_one(self):
        pass

    @staticmethod
    def static_method():
        os.path.join("a", "b")
"""


def test_python_extracts_symbols(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    names = [s.name for s in result.symbols]
    assert "top_level_func" in names
    assert "async_func" in names
    assert "MyClass" in names
    assert "method_one" in names
    assert "__init__" in names


def test_python_function_kinds(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    sym_map = {s.name: s for s in result.symbols}
    assert sym_map["top_level_func"].kind == SymbolKind.FUNCTION
    assert sym_map["MyClass"].kind == SymbolKind.CLASS
    assert sym_map["method_one"].kind == SymbolKind.METHOD


def test_python_method_has_parent(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    sym_map = {s.name: s for s in result.symbols}
    assert sym_map["method_one"].parent_name == "MyClass"
    assert sym_map["__init__"].parent_name == "MyClass"


def test_python_extracts_imports(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    modules = [imp.module_string for imp in result.imports]
    assert "os" in modules
    assert "sys" in modules
    assert "os.path" in modules
    # Relative imports
    relative = [imp for imp in result.imports if imp.is_relative]
    assert len(relative) >= 1


def test_python_import_names(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    from_os_path = next(
        (i for i in result.imports if i.module_string == "os.path"), None
    )
    assert from_os_path is not None
    assert "join" in from_os_path.imported_names
    assert "exists" in from_os_path.imported_names


def test_python_import_alias(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    sys_import = next((i for i in result.imports if i.module_string == "sys"), None)
    assert sys_import is not None
    assert sys_import.alias == "system"


def test_python_extracts_inheritance(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    assert len(result.inheritance) >= 1
    inh = result.inheritance[0]
    assert inh.class_name == "MyClass"
    assert "Base" in inh.base_names
    assert "Mixin" in inh.base_names


def test_python_extracts_calls(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    callee_names = [c.callee_name for c in result.calls]
    assert "top_level_func" in callee_names


def test_python_method_call(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("test.py", Language.PYTHON, PYTHON_SOURCE)
    method_calls = [c for c in result.calls if c.callee_object is not None]
    assert any(c.callee_name == "join" for c in method_calls)


# ---------------------------------------------------------------------------
# JavaScript extraction
# ---------------------------------------------------------------------------

JS_SOURCE = """\
import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import './styles.css';
const utils = require('./utils');

function fetchUser(id) {
    return axios.get('/api/users/' + id);
}

class AuthService extends BaseService {
    constructor() {
        super();
    }

    async login(email, password) {
        const result = fetchUser(email);
        return result;
    }
}

const arrow = () => {};
"""


def test_js_extracts_symbols(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("app.js", Language.JAVASCRIPT, JS_SOURCE)
    names = [s.name for s in result.symbols]
    assert "fetchUser" in names
    assert "AuthService" in names


def test_js_extracts_imports(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("app.js", Language.JAVASCRIPT, JS_SOURCE)
    modules = [i.module_string for i in result.imports]
    assert "react" in modules
    assert "axios" in modules


def test_js_relative_import(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("app.js", Language.JAVASCRIPT, JS_SOURCE)
    relative = [i for i in result.imports if i.is_relative]
    assert len(relative) >= 1


def test_js_class_inheritance(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("app.js", Language.JAVASCRIPT, JS_SOURCE)
    assert len(result.inheritance) >= 1
    inh = result.inheritance[0]
    assert inh.class_name == "AuthService"
    assert "BaseService" in inh.base_names


def test_js_function_calls(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("app.js", Language.JAVASCRIPT, JS_SOURCE)
    callee_names = [c.callee_name for c in result.calls]
    assert "fetchUser" in callee_names


# ---------------------------------------------------------------------------
# Unknown language graceful degradation
# ---------------------------------------------------------------------------


def test_unknown_language_returns_empty(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("file.xyz", Language.UNKNOWN, "some content")
    assert result.symbols == []
    assert result.imports == []
    assert result.calls == []
    assert result.inheritance == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file(extractor: GraphExtractor) -> None:
    result = extractor.extract_from_source("empty.py", Language.PYTHON, "")
    assert result.symbols == []
    assert result.imports == []


def test_syntax_error_file(extractor: GraphExtractor) -> None:
    # Tree-sitter is error-tolerant; it should not raise
    bad_python = "def foo(: broken syntax"
    result = extractor.extract_from_source("bad.py", Language.PYTHON, bad_python)
    # May or may not find symbols; should not raise
    assert isinstance(result.symbols, list)
