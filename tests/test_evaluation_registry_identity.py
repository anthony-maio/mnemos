from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.local_memory import heldout_run
from benchmarks.local_memory.contracts import HeldoutDataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl"


def _conflicting_project(kind: str, implicit: bool = False) -> HeldoutDataset:
    dataset = load_dataset(DATASET)
    session = next(item for item in dataset.heldout_sessions if item.state_pairs)
    values = list(getattr(session, kind))
    values[0] = replace(
        values[0],
        target_user_profile_id="different-profile",
        target_project_id=None if implicit else session.project_id,
    )
    changed = replace(session, **{kind: tuple(values)})
    return replace(
        dataset,
        sessions=tuple(
            changed if item.session_id == session.session_id else item for item in dataset.sessions
        ),
    )


@pytest.mark.parametrize("kind", ["events", "queries", "state_pairs"])
@pytest.mark.parametrize("implicit", [False, True])
def test_registry_rejects_conflicting_project_profiles(
    kind: str, implicit: bool, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="project.*conflicting profiles"):
        heldout_run._registry_for_dataset(_conflicting_project(kind, implicit), tmp_path)


def test_registry_rejects_conflicting_workspace_profiles(tmp_path: Path) -> None:
    dataset = load_dataset(DATASET)
    session = dataset.heldout_sessions[0]
    event = replace(
        session.events[0],
        target_project_id="extra-one",
        target_user_profile_id="profile-one",
        target_workspace_id="shared-workspace",
    )
    query = replace(
        session.queries[0],
        target_project_id="extra-two",
        target_user_profile_id="profile-two",
        target_workspace_id="shared-workspace",
    )
    changed = replace(session, events=(event,), queries=(query,), state_pairs=())
    dataset = replace(dataset, sessions=(changed,))
    with pytest.raises(ValueError, match="workspace.*conflicting profiles"):
        heldout_run._registry_for_dataset(dataset, tmp_path)


def test_registry_preserves_shared_workspace_for_consistent_profile(tmp_path: Path) -> None:
    dataset = load_dataset(DATASET)
    session = dataset.heldout_sessions[0]
    event = replace(
        session.events[0],
        target_project_id="extra-one",
        target_user_profile_id="profile-one",
        target_workspace_id="shared-workspace",
    )
    query = replace(
        session.queries[0],
        target_project_id="extra-two",
        target_user_profile_id="profile-one",
        target_workspace_id="shared-workspace",
    )
    changed = replace(session, events=(event,), queries=(query,), state_pairs=())
    registry = heldout_run._registry_for_dataset(replace(dataset, sessions=(changed,)), tmp_path)
    assert registry.workspaces["shared-workspace"].user_profile_id == "profile-one"
    for project in ("extra-one", "extra-two"):
        assert registry.projects[project].user_profile_id == "profile-one"
        assert registry.projects[project].workspace_ids == ["shared-workspace"]


def test_invalid_registry_fails_before_provider_probe_or_output_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(heldout_run, "load_dataset", lambda path: _conflicting_project("events"))

    def unexpected_probe(*args: object) -> None:
        raise AssertionError("invalid dataset reached provider discovery")

    monkeypatch.setattr(heldout_run, "_live_model_metadata", unexpected_probe)
    with pytest.raises(ValueError, match="conflicting profiles"):
        heldout_run.run_heldout(
            dataset_path=DATASET,
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            endpoint="http://unused.invalid",
            model="unused",
            embedding_model="unused",
            root=tmp_path,
        )
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("profiles", [{"misspelled"}, set()])
def test_invalid_profiles_fail_before_provider_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profiles: set[str]
) -> None:
    def unexpected_probe(*args: object) -> None:
        raise AssertionError("invalid profile selection reached provider discovery")

    monkeypatch.setattr(heldout_run, "_live_model_metadata", unexpected_probe)
    with pytest.raises(ValueError, match="evaluation profiles"):
        heldout_run.run_heldout(
            dataset_path=DATASET,
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            endpoint="http://unused.invalid",
            model="unused",
            embedding_model="unused",
            root=tmp_path,
            profile_names=profiles,
        )
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "work").exists()
