"""Validated identities and explicit membership for scoped local memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def default_registry_path() -> Path:
    from .settings import default_global_config_path

    configured = os.environ.get("MNEMOS_SCOPE_REGISTRY")
    return (
        Path(configured).expanduser()
        if configured
        else default_global_config_path().with_name("scopes.json")
    )


class ScopeContext(BaseModel):
    user_profile_id: str = Field(min_length=1)
    project_id: str | None = None
    workspace_id: str | None = None
    session_id: str = Field(min_length=1)
    host: Literal["codex", "claude-code", "hermes", "manual"]

    @field_validator("user_profile_id", "project_id", "workspace_id", "session_id")
    @classmethod
    def nonempty_identity(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Scope identities cannot be blank")
        return value


class ProjectIdentity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    root: str
    user_profile_id: str
    enabled: bool = False
    workspace_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class WorkspaceIdentity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    user_profile_id: str


def canonical_project_root(path: str | Path) -> str:
    """Worktrees share Git's common directory; unrelated clones do not."""
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("A project root must be a directory")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            root = Path(result.stdout.strip()).resolve()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return os.path.normcase(str(root))


class ScopeRegistry(BaseModel):
    """Configuration-owned identity map; runtime persistence is handled separately."""

    schema_version: int = 1
    projects: dict[str, ProjectIdentity] = Field(default_factory=dict)
    workspaces: dict[str, WorkspaceIdentity] = Field(default_factory=dict)
    profiles: list[str] = Field(default_factory=list)

    def register_profile(self, profile_id: str) -> str:
        normalized = profile_id.strip()
        if not normalized:
            raise ValueError("A user profile identity is required")
        if normalized not in self.profiles:
            self.profiles.append(normalized)
        return normalized

    def register_project(
        self, path: str | Path, *, user_profile_id: str, enabled: bool = False
    ) -> ProjectIdentity:
        user_profile_id = self.register_profile(user_profile_id)
        root = canonical_project_root(path)
        for project in self.projects.values():
            if project.user_profile_id == user_profile_id and root in [
                project.root,
                *project.aliases,
            ]:
                return project
        project = ProjectIdentity(root=root, user_profile_id=user_profile_id, enabled=enabled)
        self.projects[project.id] = project
        return project

    def register_workspace(self, name: str, *, user_profile_id: str) -> WorkspaceIdentity:
        if not name.strip() or not user_profile_id.strip():
            raise ValueError("Workspace name and profile are required")
        user_profile_id = self.register_profile(user_profile_id)
        workspace = WorkspaceIdentity(name=name, user_profile_id=user_profile_id)
        self.workspaces[workspace.id] = workspace
        return workspace

    def add_membership(self, project_id: str, workspace_id: str) -> None:
        project, workspace = self.projects[project_id], self.workspaces[workspace_id]
        if project.user_profile_id != workspace.user_profile_id:
            raise ValueError("Workspace membership cannot cross user profiles")
        if workspace_id not in project.workspace_ids:
            project.workspace_ids.append(workspace_id)

    def add_alias(self, project_id: str, path: str | Path) -> None:
        project = self.projects[project_id]
        root = canonical_project_root(path)
        for other in self.projects.values():
            if (
                other.id != project_id
                and other.user_profile_id == project.user_profile_id
                and root in [other.root, *other.aliases]
            ):
                raise ValueError("Alias already belongs to another project")
        if root != project.root and root not in project.aliases:
            project.aliases.append(root)

    def validate_context(self, context: ScopeContext) -> ScopeContext:
        known_profiles = (
            set(self.profiles)
            | {item.user_profile_id for item in self.projects.values()}
            | {item.user_profile_id for item in self.workspaces.values()}
        )
        if context.user_profile_id not in known_profiles:
            raise ValueError("User profile is not registered")
        if context.project_id is not None:
            project = self.projects.get(context.project_id)
            if (
                project is None
                or not project.enabled
                or project.user_profile_id != context.user_profile_id
            ):
                raise ValueError("Project is not enabled for this user profile")
            if (
                context.workspace_id is not None
                and context.workspace_id not in project.workspace_ids
            ):
                raise ValueError("Workspace membership is required")
        if context.workspace_id is not None:
            workspace = self.workspaces.get(context.workspace_id)
            if workspace is None or workspace.user_profile_id != context.user_profile_id:
                raise ValueError("Workspace membership is invalid for this profile")
        return context

    def resolve(
        self,
        path: str | Path,
        *,
        user_profile_id: str,
        session_id: str,
        host: Literal["codex", "claude-code", "hermes", "manual"],
        workspace_id: str | None = None,
    ) -> ScopeContext:
        root = canonical_project_root(path)
        for project in self.projects.values():
            if project.user_profile_id == user_profile_id and root in [
                project.root,
                *project.aliases,
            ]:
                return self.validate_context(
                    ScopeContext(
                        user_profile_id=user_profile_id,
                        project_id=project.id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        host=host,
                    )
                )
        raise ValueError("Project is not enabled; assign its identity before capture")

    def save(self, path: str | Path) -> None:
        from .service import _write_private_json

        _write_private_json(Path(path), self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: str | Path) -> ScopeRegistry:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported scope registry schema")
        return cls.model_validate(data)


def scope_allows(
    metadata: Mapping[str, Any], context: ScopeContext, allowed_scopes: Sequence[str]
) -> bool:
    """Unassigned legacy metadata is never implicitly shared global memory."""
    if metadata.get("user_profile_id") != context.user_profile_id:
        return False
    scope = metadata.get("scope")
    if scope not in allowed_scopes:
        return False
    if scope == "global":
        return True
    expected = (
        context.project_id
        if scope == "project"
        else context.workspace_id if scope == "workspace" else None
    )
    return expected is not None and metadata.get("scope_id") == expected


def context_metadata(context: ScopeContext, scope: str) -> dict[str, Any]:
    if scope not in ("project", "workspace", "global"):
        raise ValueError("Invalid memory scope")
    identity = (
        context.project_id
        if scope == "project"
        else context.workspace_id if scope == "workspace" else None
    )
    if scope != "global" and identity is None:
        raise ValueError(f"A {scope} identity and membership are required")
    return {
        **context.model_dump(),
        "scope": scope,
        "scope_id": identity,
    }


def interaction_partition(metadata: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Legacy callers share state only within their explicit scope and session."""
    return (
        str(metadata.get("user_profile_id", "legacy")),
        str(metadata.get("scope", "project")),
        str(metadata.get("scope_id", "default")),
        str(metadata.get("session_id", "legacy")),
    )
