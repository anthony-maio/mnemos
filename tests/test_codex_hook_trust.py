"""Native trust acceptance must reject altered or merely configured hooks."""

import copy
from pathlib import Path

import pytest

from benchmarks.native_hosts.codex_hook_trust import validate_inventory


def inventory(tmp_path: Path):
    config = tmp_path / "home/config.toml"
    project = tmp_path / "project"
    expected = {}
    hooks = []
    for index, event in enumerate(
        ("SessionStart", "UserPromptSubmit", "PreCompact", "PostCompact", "Stop", "SessionEnd")
    ):
        group = {"hooks": [{"type": "command", "command": "reviewed command", "timeout": 3}]}
        expected[event] = [group]
        hooks.append(
            dict(
                eventName=event[0].lower() + event[1:],
                handlerType="command",
                command="reviewed command",
                timeoutSec=3,
                matcher=None,
                additionalContextLimit=None,
                source="user",
                sourcePath=str(config),
                enabled=True,
                isManaged=False,
                key=f"hook-{index}",
                currentHash=f"digest-{index}",
                trustStatus="trusted",
            )
        )
    return (
        {"data": [{"cwd": str(project), "errors": [], "warnings": [], "hooks": hooks}]},
        expected,
        project,
        config,
    )


def check(tmp_path, alter=None, **kwargs):
    result, expected, project, config = inventory(tmp_path)
    if alter:
        alter(result)
    return validate_inventory(
        result, project=project, config_path=config, expected_hooks=expected, **kwargs
    )


def test_exact_persisted_trust_is_accepted(tmp_path):
    result = check(tmp_path, require_trusted=True)
    assert result["status"] == "trusted" and len(result["current_hashes"]) == 6
    assert check(tmp_path, require_trusted=True, expected_hashes=result["current_hashes"]) == result


@pytest.mark.parametrize("status", ["untrusted", "modified"])
def test_configuration_and_changed_definition_do_not_establish_trust(tmp_path, status):
    alter = lambda value: value["data"][0]["hooks"][0].update(trustStatus=status)
    assert check(tmp_path, alter)["status"] == "review_required"
    with pytest.raises(ValueError, match="review is still required"):
        check(tmp_path, alter, require_trusted=True)


@pytest.mark.parametrize(
    "change",
    [
        {"command": "unreviewed command"},
        {"matcher": ".*"},
        {"timeoutSec": 900},
        {"additionalContextLimit": 5000},
        {"source": "project"},
        {"sourcePath": "elsewhere/config.toml"},
        {"isManaged": True},
        {"isManaged": 0},
        {"enabled": 1},
        {"enabled": False},
        {"async": True},
        {"handlerType": "mcpTool"},
        {"key": ""},
        {"key": "hook-1"},
        {"currentHash": ""},
        {"trustStatus": "managed"},
        {"trustStatus": None},
        {"eventName": "Stop"},
        {"eventName": "UnreviewedEvent"},
    ],
)
def test_metadata_must_match_the_reviewed_definition(tmp_path, change):
    with pytest.raises(ValueError):
        check(
            tmp_path,
            lambda value: value["data"][0]["hooks"][0].update(change),
            require_trusted=True,
        )


@pytest.mark.parametrize(
    "alter",
    [
        lambda value: value["data"].clear(),
        lambda value: value["data"].append(copy.deepcopy(value["data"][0])),
        lambda value: value["data"][0].update(cwd="elsewhere"),
        lambda value: value["data"][0].update(errors=["configuration error"]),
        lambda value: value["data"][0]["hooks"].pop(),
        lambda value: value["data"][0]["hooks"].append(copy.deepcopy(value["data"][0]["hooks"][0])),
    ],
)
def test_missing_foreign_or_additional_hooks_fail_closed(tmp_path, alter):
    with pytest.raises(ValueError):
        check(tmp_path, alter, require_trusted=True)


def test_changed_hash_after_preparation_requires_new_review(tmp_path):
    expected = check(tmp_path)["current_hashes"]
    with pytest.raises(ValueError, match="changed since review"):
        check(
            tmp_path,
            lambda value: value["data"][0]["hooks"][0].update(currentHash="different"),
            require_trusted=True,
            expected_hashes=expected,
        )
