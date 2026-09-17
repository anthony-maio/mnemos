from pathlib import Path
import subprocess

import pytest

from mnemos.scopes import ScopeContext, ScopeRegistry, scope_allows
from mnemos.types import Interaction
from mnemos.modules.surprisal import SurprisalGate
from mnemos.config import SurprisalConfig
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore


def test_same_directory_name_is_not_a_shared_project(tmp_path: Path) -> None:
    first = tmp_path / "one" / "app"
    second = tmp_path / "two" / "app"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    registry = ScopeRegistry()
    a = registry.register_project(first, user_profile_id="local", enabled=True)
    b = registry.register_project(second, user_profile_id="local", enabled=True)
    assert a.id != b.id
    destination = tmp_path / "registry.json"
    registry.save(destination)
    resumed = ScopeRegistry.load(destination)
    assert (
        resumed.resolve(first, user_profile_id="local", session_id="new", host="codex").project_id
        == a.id
    )


def test_workspace_and_profile_membership_are_enforced(tmp_path: Path) -> None:
    registry = ScopeRegistry()
    project = registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    workspace = registry.register_workspace("work", user_profile_id="local")
    with pytest.raises(ValueError, match="membership"):
        registry.resolve(
            tmp_path,
            user_profile_id="local",
            session_id="s",
            host="hermes",
            workspace_id=workspace.id,
        )
    registry.add_membership(project.id, workspace.id)
    ctx = registry.resolve(
        tmp_path, user_profile_id="local", session_id="s", host="hermes", workspace_id=workspace.id
    )
    assert scope_allows(
        {"scope": "workspace", "scope_id": workspace.id, "user_profile_id": "local"},
        ctx,
        ("workspace",),
    )
    assert not scope_allows({"scope": "global", "user_profile_id": "other"}, ctx, ("global",))
    assert not scope_allows({}, ctx, ("project", "workspace", "global"))
    assert not scope_allows(
        {"scope": "project", "scope_id": "foreign", "user_profile_id": "local"},
        ctx,
        ("project", "global"),
    )


def test_unknown_or_disabled_project_does_not_become_default(tmp_path: Path) -> None:
    registry = ScopeRegistry()
    with pytest.raises(ValueError, match="enabled"):
        registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    registry.register_project(tmp_path, user_profile_id="local")
    with pytest.raises(ValueError, match="enabled"):
        registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")


def test_worktrees_share_project_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )
    worktree = tmp_path / "branch"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", "other", str(worktree)],
        check=True,
        capture_output=True,
    )
    registry = ScopeRegistry()
    original = registry.register_project(repo, user_profile_id="local", enabled=True)
    assert (
        registry.register_project(worktree, user_profile_id="local", enabled=True).id == original.id
    )


@pytest.mark.asyncio
async def test_prediction_never_uses_another_scope_or_session() -> None:
    llm = MockLLMProvider(record_calls=True)
    gate = SurprisalGate(
        llm, SimpleEmbeddingProvider(), InMemoryStore(), SurprisalConfig(threshold=0.0)
    )
    for project, session, text in [
        ("a", "s1", "Project A has the private CANARY_ALPHA convention."),
        ("b", "s1", "Project B has the private CANARY_BETA convention."),
        ("b", "s2", "What is our current repository convention?"),
    ]:
        await gate.process(
            Interaction(
                role="user",
                content=text,
                metadata={"scope": "project", "scope_id": project, "session_id": session},
            )
        )
    predictions = [p for p in llm.calls if "predict what" in p.lower()]
    assert "CANARY_ALPHA" not in predictions[1]
    assert "CANARY_ALPHA" not in predictions[2]
    assert "CANARY_BETA" not in predictions[2]


def test_scope_context_rejects_empty_identity() -> None:
    with pytest.raises(ValueError):
        ScopeContext(user_profile_id="", session_id="s", host="codex")


@pytest.mark.asyncio
async def test_affective_fallback_is_partitioned_by_profile_project_and_session() -> None:
    from mnemos.modules.affective import AffectiveRouter
    from mnemos.types import CognitiveState

    router = AffectiveRouter(
        MockLLMProvider(responses={"valence": "-0.8, 0.9, 0.7"}), SimpleEmbeddingProvider()
    )
    metadata = {
        "user_profile_id": "alice",
        "scope": "project",
        "scope_id": "repo",
        "session_id": "s1",
    }
    await router.classify_state(
        Interaction(role="user", content="This incident is urgent", metadata=metadata)
    )
    assert router.get_current_state(metadata).arousal == 0.9
    for override in ({"user_profile_id": "bob"}, {"scope_id": "other"}, {"session_id": "s2"}):
        assert router.get_current_state({**metadata, **override}) == CognitiveState(
            valence=0, arousal=0.5, complexity=0.5
        )


@pytest.mark.asyncio
async def test_other_sessions_do_not_evict_prediction_history() -> None:
    llm = MockLLMProvider(record_calls=True)
    gate = SurprisalGate(
        llm, SimpleEmbeddingProvider(), InMemoryStore(), SurprisalConfig(history_window=2)
    )
    own = {"scope": "project", "scope_id": "a", "session_id": "s"}
    gate.add_to_history(Interaction(role="user", content="OWN_HISTORY_CANARY", metadata=own))
    for i in range(5):
        gate.add_to_history(
            Interaction(
                role="user",
                content="OTHER_HISTORY_CANARY",
                metadata={**own, "session_id": f"other-{i}"},
            )
        )
    await gate.process(Interaction(role="user", content="What should we do next?", metadata=own))
    prompt = next(p for p in llm.calls if "predict what" in p.lower())
    assert "OWN_HISTORY_CANARY" in prompt
    assert "OTHER_HISTORY_CANARY" not in prompt


@pytest.mark.asyncio
async def test_consolidation_never_combines_user_profiles() -> None:
    from mnemos.modules.sleep import SleepDaemon
    from mnemos.config import SleepConfig

    llm = MockLLMProvider(record_calls=True)
    store = InMemoryStore()
    daemon = SleepDaemon(store, SleepConfig(min_episodes_before_consolidation=1))
    for profile in ("ALPHA", "BETA"):
        daemon.add_episode(
            Interaction(
                role="user",
                content=f"{profile}_PRIVATE_PREFERENCE",
                metadata={"user_profile_id": profile, "scope": "global", "session_id": "same"},
            )
        )
    await daemon.consolidate(llm, SimpleEmbeddingProvider())
    prompts = [p for p in llm.calls if "extract permanent facts" in p.lower()]
    assert len(prompts) == 2
    assert all(not ("ALPHA_PRIVATE" in p and "BETA_PRIVATE" in p) for p in prompts)
    assert {c.metadata.get("user_profile_id") for c in store.get_all()} == {"ALPHA", "BETA"}


@pytest.mark.asyncio
async def test_engine_scopes_cannot_be_spoofed_in_metadata(tmp_path: Path) -> None:
    from mnemos.engine import MnemosEngine
    from mnemos.config import MnemosConfig

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    ctx = registry.resolve(tmp_path, user_profile_id="alice", session_id="s", host="codex")
    engine = MnemosEngine(
        llm=MockLLMProvider(),
        scope_registry=registry,
        config=MnemosConfig(surprisal=SurprisalConfig(threshold=0, min_content_length=0)),
    )
    result = await engine.process(
        Interaction(
            role="user",
            content="Use SQLite in this project",
            metadata={"user_profile_id": "bob", "scope_id": "foreign", "session_id": "forged"},
        ),
        context=ctx,
    )
    assert result.chunk is not None
    assert result.chunk.metadata["user_profile_id"] == "alice"
    assert result.chunk.metadata["scope_id"] == ctx.project_id
    assert result.chunk.metadata["session_id"] == "s"
    with pytest.raises(ValueError, match="scope"):
        await engine.process(
            Interaction(role="user", content="Should not be stored"),
            context=ctx,
            scope_id="foreign",
        )
    with pytest.raises(ValueError, match="membership"):
        await engine.process(
            Interaction(role="user", content="Should not be stored"), context=ctx, scope="workspace"
        )


@pytest.mark.asyncio
async def test_explicit_profile_retrieval_excludes_foreign_and_unassigned(tmp_path: Path) -> None:
    from mnemos.engine import MnemosEngine
    from mnemos.types import MemoryChunk

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    registry.register_project(tmp_path, user_profile_id="bob", enabled=True)
    ctx = registry.resolve(tmp_path, user_profile_id="alice", session_id="s", host="codex")
    store = InMemoryStore()
    embedder = SimpleEmbeddingProvider()
    for identifier, metadata in [
        ("own", {"scope": "global", "user_profile_id": "alice"}),
        ("foreign", {"scope": "global", "user_profile_id": "bob"}),
        ("unassigned", {}),
    ]:
        store.store(
            MemoryChunk(
                id=identifier,
                content="SQLite convention",
                embedding=embedder.embed("SQLite convention"),
                metadata=metadata,
            )
        )
    engine = MnemosEngine(
        llm=MockLLMProvider(), embedder=embedder, store=store, scope_registry=registry
    )
    found = await engine.retrieve("SQLite convention", top_k=5, reconsolidate=False, context=ctx)
    assert [c.id for c in found] == ["own"]
    assert not await engine.retrieve("SQLite convention", top_k=5, reconsolidate=False)
