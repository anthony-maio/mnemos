"""Real Windows ACL checks for private runtime and outbox paths."""

from __future__ import annotations

import json
import ctypes
import os
from pathlib import Path
import subprocess

import pytest

from mnemos.service import _ProcessLock, _protect_private_path

pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires Windows file ACLs")


def acl(path: Path, *, grant_everyone: bool = False) -> dict:
    if grant_everyone:
        inheritance = "(OI)(CI)" if path.is_dir() else ""
        subprocess.run(
            ["icacls.exe", str(path), "/grant", f"*S-1-1-0:{inheritance}RX"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    command = """
$ErrorActionPreference = 'Stop'
$target = $env:MNEMOS_TEST_ACL_PATH
$acl = Get-Acl -LiteralPath $target
[ordered]@{
    protected = $acl.AreAccessRulesProtected
    current_sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | ForEach-Object {
        [ordered]@{sid=$_.IdentityReference.Value; inherited=$_.IsInherited;
            rights=[int]$_.FileSystemRights; flags=[int]$_.InheritanceFlags;
            allow=($_.AccessControlType -eq 'Allow')}
    })
} | ConvertTo-Json -Depth 4 -Compress
"""
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        env={
            **{k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"},
            "MNEMOS_TEST_ACL_PATH": str(path),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def assert_private(path: Path, *, directory: bool) -> None:
    observed = acl(path)
    assert observed["protected"] is True
    assert observed["rules"] == [
        {
            "sid": observed["current_sid"],
            "inherited": False,
            "rights": 0x1F01FF,
            "flags": 3 if directory else 0,
            "allow": True,
        }
    ]


@pytest.mark.parametrize("directory", [False, True])
def test_protection_removes_inherited_access_and_blocks_parent_changes(tmp_path, directory):
    parent = tmp_path / "parent"
    parent.mkdir()
    acl(parent, grant_everyone=True)
    target = parent / "private"
    target.mkdir() if directory else target.write_text("synthetic", encoding="utf-8")
    assert any(rule["sid"] == "S-1-1-0" for rule in acl(target)["rules"])

    _protect_private_path(target)

    assert_private(target, directory=directory)
    acl(parent, grant_everyone=True)
    assert_private(target, directory=directory)


@pytest.mark.parametrize(
    ("api", "result"),
    [
        ("GetNamedSecurityInfoW", 5),
        ("GetNamedSecurityInfoW", 0),
        ("GetSecurityDescriptorControl", 0),
        ("GetAclInformation", 0),
    ],
)
def test_acl_query_failure_falls_back_to_permission_repair(tmp_path, monkeypatch, api, result):
    target = tmp_path / "private"
    target.touch()
    _protect_private_path(target)
    acl(target, grant_everyone=True)
    load_dll = ctypes.WinDLL
    queries = []

    def tracked_dll(name, **kwargs):
        dll = load_dll(name, **kwargs)
        if name == "advapi32":

            def denied(*args):
                queries.append(api)
                return result  # A failed query or missing descriptor cannot prove privacy.

            setattr(dll, api, denied)
        return dll

    monkeypatch.setattr(ctypes, "WinDLL", tracked_dll)
    _protect_private_path(target)
    assert queries == [api]
    assert_private(target, directory=False)


def test_process_lock_repairs_legacy_inherited_permissions(tmp_path):
    parent = tmp_path / "runtime"
    parent.mkdir()
    acl(parent, grant_everyone=True)
    path = parent / "outbox.lock"
    path.touch()
    lock = _ProcessLock(path)
    assert lock.acquire()
    try:
        assert_private(parent, directory=True)
        assert_private(path, directory=False)
    finally:
        lock.release()


@pytest.mark.parametrize("directory", [False, True])
def test_private_acl_is_checked_without_rewriting_and_drift_is_repaired(
    tmp_path, monkeypatch, directory
):
    target = tmp_path / "private"
    target.mkdir() if directory else target.touch()
    writes = []
    load_dll = ctypes.WinDLL

    def tracked_dll(name, **kwargs):
        dll = load_dll(name, **kwargs)
        if name == "advapi32":
            original = dll.SetNamedSecurityInfoW

            def tracked_set(*args):
                original.argtypes = tracked_set.argtypes
                original.restype = tracked_set.restype
                writes.append(args[0])
                return original(*args)

            dll.SetNamedSecurityInfoW = tracked_set
        return dll

    monkeypatch.setattr(ctypes, "WinDLL", tracked_dll)
    _protect_private_path(target)
    assert len(writes) == 1
    assert_private(target, directory=directory)

    _protect_private_path(target)
    assert len(writes) == 1, "an unchanged private ACL must not trigger recursive propagation"

    acl(target, grant_everyone=True)
    _protect_private_path(target)
    assert len(writes) == 2, "ACL drift must be repaired, not hidden by a path cache"
    assert_private(target, directory=directory)
