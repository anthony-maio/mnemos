"""
tests/test_docs_consistency.py — Keep docs aligned with runtime behavior.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = ROOT / "mnemos" / "mcp_server.py"
RUNTIME_PATH = ROOT / "mnemos" / "runtime.py"
README_PATH = ROOT / "README.md"
MCP_INTEGRATION_PATH = ROOT / "docs" / "MCP_INTEGRATION.md"
MCP_CONTRACT_PATH = ROOT / "docs" / "mcp-transport-contract.md"
ARCHITECTURE_PATH = ROOT / "ARCHITECTURE.md"
REQUIRED_OSS_DOCS = [
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "CODE_OF_CONDUCT.md",
    ROOT / "SUPPORT.md",
]

MNEMOS_ENV_RE = re.compile(r"\bMNEMOS_[A-Z0-9_]+\b")
MNEMOS_RESOURCE_RE = re.compile(r"\bmnemos://[a-z0-9_-]+\b", re.IGNORECASE)


def _node_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_env_vars_from_source(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    vars_found: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "getenv"
        ):
            name = _node_str(node.args[0]) if node.args else None
            if name and name.startswith("MNEMOS_"):
                vars_found.add(name)
            continue

        if isinstance(node.func, ast.Name) and node.func.id == "resolve_env_value":
            name = _node_str(node.args[0]) if node.args else None
            if name and name.startswith("MNEMOS_"):
                vars_found.add(name)

            for keyword in node.keywords:
                if keyword.arg != "aliases" or not isinstance(keyword.value, (ast.Tuple, ast.List)):
                    continue
                for alias in keyword.value.elts:
                    alias_name = _node_str(alias)
                    if alias_name and alias_name.startswith("MNEMOS_"):
                        vars_found.add(alias_name)

    return vars_found


def _collect_resources_from_server(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    resources: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "resource"
            ):
                uri = _node_str(decorator.args[0]) if decorator.args else None
                if uri and uri.startswith("mnemos://"):
                    resources.add(uri)

    return resources


def _collect_tokens(
    paths: list[Path],
    pattern: re.Pattern[str],
    *,
    normalize_lower: bool = False,
) -> set[str]:
    tokens: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        if normalize_lower:
            tokens.update(match.lower() for match in matches)
        else:
            tokens.update(matches)
    return tokens


def test_runtime_env_vars_are_documented() -> None:
    runtime_vars = _collect_env_vars_from_source(RUNTIME_PATH)
    server_vars = _collect_env_vars_from_source(MCP_SERVER_PATH)
    code_vars = runtime_vars | server_vars

    documented_vars = _collect_tokens(
        [README_PATH, MCP_INTEGRATION_PATH],
        MNEMOS_ENV_RE,
    )

    undocumented = sorted(var for var in code_vars if var not in documented_vars)
    assert undocumented == [], (
        "Document these env vars in README.md or docs/MCP_INTEGRATION.md: " f"{undocumented}"
    )


def test_mcp_resources_match_documented_contract() -> None:
    server_resources = {uri.lower() for uri in _collect_resources_from_server(MCP_SERVER_PATH)}
    documented_resources = _collect_tokens(
        [MCP_CONTRACT_PATH, README_PATH, MCP_INTEGRATION_PATH, MCP_SERVER_PATH],
        MNEMOS_RESOURCE_RE,
        normalize_lower=True,
    )

    missing_from_docs = sorted(uri for uri in server_resources if uri not in documented_resources)
    unknown_in_docs = sorted(uri for uri in documented_resources if uri not in server_resources)

    assert missing_from_docs == [], (
        "Document these MCP resources in docs/mcp-transport-contract.md: " f"{missing_from_docs}"
    )
    assert unknown_in_docs == [], (
        "Docs mention MCP resources not exposed by the server: " f"{unknown_in_docs}"
    )


def test_backend_status_labels_are_explicit() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    architecture = ARCHITECTURE_PATH.read_text(encoding="utf-8")

    assert "QdrantStore" in readme
    assert "QdrantStore" in architecture
    assert re.search(r"Neo4j`?\s+\(planned\)", readme) is not None
    assert re.search(r"Neo4j`?\s+\(planned\)", architecture) is not None


def test_required_oss_docs_exist() -> None:
    missing = [path.name for path in REQUIRED_OSS_DOCS if not path.exists()]
    assert missing == [], f"Add the standard OSS trust docs before release: {missing}"
