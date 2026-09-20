"""Scope-bound Hermes capture identity shared by adapter and service."""

import hashlib
import json
import re

from .scopes import ScopeContext


def is_summary_source_ref(source: str) -> bool:
    """Only a durable native carrier can have a stable summary receipt."""
    return re.fullmatch(r"provisional:v2:summary:[1-9][0-9]*", source) is not None


def canonical_native_ref(source: str) -> str:
    """Classification changes must conflict with the same physical occurrence."""
    return f"transcript:v2:{source.rsplit(':', 1)[1]}" if is_summary_source_ref(source) else source


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
