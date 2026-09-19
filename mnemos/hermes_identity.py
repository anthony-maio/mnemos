"""Scope-bound Hermes capture identity shared by adapter and service."""

import hashlib
import json

from .scopes import ScopeContext


def scoped_event_id(context: ScopeContext, source_ref: str, role: str, content: str) -> str:
    material = json.dumps(
        [
            context.user_profile_id,
            context.project_id,
            context.workspace_id,
            context.session_id,
            role,
            source_ref,
            content,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "hermes:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
