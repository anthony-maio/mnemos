"""Explicit profile isolation for the reviewed OAuth Claude print-mode binary."""

import hashlib
from pathlib import Path
from typing import Mapping

REVIEWED_SHA256 = "83397a6a029c7da663fb1ce27211e05174a3546d8b151e42451bf4590b8343d7"
REMOVED = frozenset(
    {
        "CLAUDE_BG_AUTH_SNAPSHOT_PATH",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        "CLAUDE_CODE_SIMPLE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_SECURESTORAGE_CONFIG_DIR",
        "ANTHROPIC_CONFIG_DIR",
    }
)
OVERRIDDEN = frozenset(
    {
        "CLAUDE_CONFIG_DIR",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
        "DISABLE_AUTOUPDATER",
        "ENABLE_CLAUDEAI_MCP_SERVERS",
        "MCP_CONNECTION_NONBLOCKING",
    }
)


def environment(base: Mapping[str, str], home: Path, root: Path) -> dict[str, str]:
    """Require an existing private home, then override every profile selector.

    The caller still owns ACL protection, provider-selector cleanup, credential
    copying, subprocess ownership, and cleanup. No credential is read here.
    """
    home = home.resolve(strict=True)
    root = root.resolve(strict=True)
    if home == root or not home.is_relative_to(root) or not home.is_dir():
        raise ValueError("Claude home must be a private child of the disposable root")
    result = {k: v for k, v in base.items() if k.upper() not in REMOVED | OVERRIDDEN}
    result.update(
        {
            "CLAUDE_CONFIG_DIR": str(home),
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "DISABLE_AUTOUPDATER": "1",
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "MCP_CONNECTION_NONBLOCKING": "false",
        }
    )
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        location = home / key.lower()
        if location.exists() and not location.resolve().is_relative_to(home):
            raise ValueError("Claude profile path escapes private home")
        location.mkdir(exist_ok=True)
        result[key] = str(location)
    profile = Path(result["USERPROFILE"])
    result["HOMEDRIVE"] = profile.drive
    result["HOMEPATH"] = str(profile)[len(profile.drive) :]
    return result


def verify_binary(executable: Path) -> None:
    if hashlib.sha256(executable.read_bytes()).hexdigest() != REVIEWED_SHA256:
        raise ValueError("Claude binary differs from the reviewed isolation implementation")
