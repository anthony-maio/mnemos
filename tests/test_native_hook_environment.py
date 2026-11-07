"""Reproduce Windows hidden-drive environment inheritance through Python."""

import json
import subprocess
import sys

import pytest

from tests.test_native_hook_launcher import BINARY, http, isolated, reply, save_descriptor, server
from tests.test_native_hook_profile import digest, profile

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not BINARY, reason="Windows native launcher required"
)


@pytest.mark.parametrize("host", ["codex", "claude-code"])
@pytest.mark.parametrize("profile_mode", [False, True])
def test_python_child_inherits_hidden_drive_entry(isolated, host, profile_mode):
    root, env = isolated
    path, _, effective = profile(root, env, host=host)
    command = [BINARY, host, "Stop"]
    if profile_mode:
        command += ["--profile", str(path), "--profile-sha256", digest(path)]
    else:
        command += ["--python", sys.executable]
        effective = env.copy()
    child = (
        "import os,subprocess,sys\n"
        "assert '=Q:' not in os.environ\n"
        "p=subprocess.run(sys.argv[1:],input=sys.stdin.buffer.read(),capture_output=True,timeout=15)\n"
        "sys.stdout.buffer.write(p.stdout)\n"
        "sys.stderr.buffer.write(p.stderr)\n"
        "sys.exit(p.returncode)\n"
    )
    # Python omits this entry from os.environ but an env=None child retains it.
    env["=Q:"] = "Q:\\synthetic-hidden-drive"
    with server(lambda request: http(reply(request))) as (port, requests):
        save_descriptor(effective, port, host=host)
        result = subprocess.run(
            [sys.executable, "-I", "-c", child, *command],
            cwd=root,
            env=env,
            input=b'{"last_assistant_message":"synthetic hidden drive"}',
            capture_output=True,
            timeout=20,
        )
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout) == {"continue": True}
    assert len(requests) == 1
    assert not (root / "guarded.input").exists()


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_ordinary_environment_mismatch_still_declines(isolated, host):
    root, env = isolated
    path, _, effective = profile(root, env, host=host)
    env["NATIVE_TEST_ORDINARY_SELECTOR"] = "different-from-worker"
    with server(lambda request: http(reply(request))) as (port, requests):
        save_descriptor(effective, port, host=host)
        result = subprocess.run(
            [BINARY, host, "Stop", "--profile", str(path), "--profile-sha256", digest(path)],
            cwd=root,
            env=env,
            input=b'{"last_assistant_message":"ordinary selector"}',
            capture_output=True,
            timeout=20,
        )
    assert result.returncode == 0 and not result.stderr and not requests
    assert json.loads(result.stdout)["guarded_fallback"] is True
