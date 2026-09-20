"""Read-only source-event association diagnostics for development fixtures.

This module deliberately does not use graph labels, embeddings, or text
similarity.  It opens an existing SQLite database in read-only mode, takes one
transactional snapshot, and emits deterministic fact pairs only when the
pairs are supported by retained ``fact_evidence.source_event_id`` rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

SCHEMA_VERSION = "source-association-diagnostic-v1"
_VALID_SCOPES = {"project", "workspace", "global"}
_VALID_EVENT_ROLES = {"user", "assistant", "tool", "system"}


@dataclass(frozen=True)
class _ScopeIdentity:
    scope: str
    scope_id: str | None
    profile: str | None
    project_id: str | None
    workspace_id: str | None

    def as_json(self) -> dict[str, str | None]:
        return {
            "scope": self.scope,
            "scope_id": self.scope_id,
            "user_profile_id": self.profile,
        }


@dataclass
class _FactObservation:
    fact_id: str
    content: str
    version: int
    scope: _ScopeIdentity | None
    evidence_ids: list[str]
    reasons: set[str]


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("identifier is not a string")
    value = value.strip()
    return value or None


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _metadata(raw: Any) -> dict[str, Any] | None:
    try:
        decoded = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _scope_identity(
    *,
    metadata: dict[str, Any] | None,
    column_scope: Any = None,
    column_scope_id: Any = None,
    column_profile: Any = None,
    require_scope: bool,
) -> tuple[_ScopeIdentity | None, str | None]:
    """Normalize storage scope fields and reject contradictory duplicates."""

    if metadata is None:
        return None, "malformed_scope_metadata"
    try:
        column_profile_id = _clean_optional(column_profile)
        metadata_profile = _clean_optional(metadata.get("user_profile_id"))
        legacy_profile = _clean_optional(metadata.get("profile_id"))
    except ValueError:
        return None, "malformed_scope_metadata"
    if (
        metadata_profile is not None
        and legacy_profile is not None
        and metadata_profile != legacy_profile
    ):
        return None, "profile_metadata_mismatch"
    profile = metadata_profile or legacy_profile
    if column_profile_id is not None and profile is not None and column_profile_id != profile:
        return None, "profile_metadata_mismatch"
    profile = column_profile_id or profile

    raw_metadata_scope = metadata.get("scope")
    if raw_metadata_scope is not None and not isinstance(raw_metadata_scope, str):
        return None, "malformed_scope_metadata"
    metadata_scope = (
        raw_metadata_scope.strip().lower() if isinstance(raw_metadata_scope, str) else None
    )
    raw_column_scope = column_scope
    if raw_column_scope is not None and not isinstance(raw_column_scope, str):
        return None, "malformed_scope_metadata"
    storage_scope = raw_column_scope.strip().lower() if isinstance(raw_column_scope, str) else None
    if metadata_scope is not None and storage_scope is not None and metadata_scope != storage_scope:
        return None, "scope_metadata_mismatch"
    scope = metadata_scope or storage_scope
    if scope is None:
        if require_scope:
            return None, "missing_scope"
        scope = "global"
    if scope not in _VALID_SCOPES:
        return None, "invalid_scope"

    try:
        storage_scope_id = _clean_optional(column_scope_id)
        metadata_scope_id = _clean_optional(metadata.get("scope_id"))
        project_id = _clean_optional(metadata.get("project_id"))
        workspace_id = _clean_optional(metadata.get("workspace_id"))
    except ValueError:
        return None, "malformed_scope_metadata"

    if (
        storage_scope_id is not None
        and metadata_scope_id is not None
        and storage_scope_id != metadata_scope_id
    ):
        return None, "scope_id_metadata_mismatch"
    scope_id = storage_scope_id or metadata_scope_id
    if scope == "project":
        if project_id is not None and scope_id is not None and project_id != scope_id:
            return None, "scope_id_metadata_mismatch"
        scope_id = project_id or scope_id
        if scope_id is None:
            return None, "missing_scope_id"
    elif scope == "workspace":
        if workspace_id is not None and scope_id is not None and workspace_id != scope_id:
            return None, "scope_id_metadata_mismatch"
        scope_id = workspace_id or scope_id
        if scope_id is None:
            return None, "missing_scope_id"
    else:
        scope_id = None

    return _ScopeIdentity(scope, scope_id, profile, project_id, workspace_id), None


def _scopes_can_connect(first: _ScopeIdentity, second: _ScopeIdentity) -> bool:
    """Match SpreadingActivation's profile and scope boundary contract."""

    if first.profile != second.profile:
        return False
    if first.scope == "global" or second.scope == "global":
        return True
    if first.scope == second.scope:
        return first.scope_id is not None and first.scope_id == second.scope_id
    if first.scope == "project" and second.scope == "workspace":
        return first.workspace_id is not None and first.workspace_id == second.scope_id
    if first.scope == "workspace" and second.scope == "project":
        return second.workspace_id is not None and second.workspace_id == first.scope_id
    return False


def _open_snapshot(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"file:{quote(path.as_posix(), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    return connection


def _relation_id(source_event_id: str, first_id: str, second_id: str) -> str:
    payload = "\0".join((source_event_id, first_id, second_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_source_associations(
    db_path: str | Path,
    *,
    fanout_limit: int = 16,
) -> dict[str, Any]:
    """Inspect actual source-event evidence relations in one SQLite snapshot.

    The returned relation list contains deterministic unordered fact pairs.
    A fact is eligible only when it is asserted, not captured passthrough,
    unsuppressed, scope-valid, and has evidence whose normalized excerpt still
    equals the current fact content.  Since ``FactEvidence`` has no version,
    an excerpt mismatch is conservatively reported as unverifiable rather than
    treated as current support.
    """

    if isinstance(fanout_limit, bool) or not isinstance(fanout_limit, int) or fanout_limit < 2:
        raise ValueError("fanout_limit must be an integer >= 2")

    connection = _open_snapshot(db_path)
    try:
        suppressed_fact_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT fact_id FROM memory_suppressions WHERE fact_id IS NOT NULL"
            )
        }
        suppressed_source_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT source_event_id FROM memory_suppressions WHERE source_event_id IS NOT NULL"
            )
        }
        rows = connection.execute("""
            SELECT fe.evidence_id, fe.fact_id, fe.source_event_id, fe.excerpt,
                   mc.content AS fact_content, mc.version AS fact_version,
                   mc.metadata AS fact_metadata, mc.user_profile_id AS fact_profile,
                   ev.id AS event_id, ev.role AS event_role, ev.source AS event_source,
                   ev.scope AS event_scope, ev.scope_id AS event_scope_id,
                   ev.user_profile_id AS event_profile, ev.metadata AS event_metadata
            FROM fact_evidence AS fe
            LEFT JOIN memory_chunks AS mc ON mc.id = fe.fact_id
            LEFT JOIN episodic_events AS ev ON ev.id = fe.source_event_id
            ORDER BY COALESCE(fe.source_event_id, ''), fe.fact_id, fe.evidence_id
            """).fetchall()

        grouped: dict[str | None, list[sqlite3.Row]] = {}
        for row in rows:
            source_event_id = row["source_event_id"]
            key = str(source_event_id) if source_event_id is not None else None
            grouped.setdefault(key, []).append(row)

        relations: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        for source_event_id in sorted(grouped, key=lambda item: "" if item is None else item):
            group_rows = grouped[source_event_id]
            fact_ids = sorted({str(row["fact_id"]) for row in group_rows})
            rejection_reasons: set[str] = set()
            if source_event_id is None:
                rejection_reasons.add("missing_source_event_id")

            event_scope: _ScopeIdentity | None = None
            event_role: str | None = None
            event_source: str | None = None
            if source_event_id is not None:
                if source_event_id in suppressed_source_ids:
                    rejection_reasons.add("suppressed_source_event")
                first_row = group_rows[0]
                if first_row["event_id"] is None:
                    rejection_reasons.add("missing_source_event")
                else:
                    raw_role = first_row["event_role"]
                    raw_source = first_row["event_source"]
                    event_role = raw_role.strip().lower() if isinstance(raw_role, str) else None
                    event_source = raw_source.strip() if isinstance(raw_source, str) else None
                    if event_role not in _VALID_EVENT_ROLES:
                        rejection_reasons.add("unsupported_event_role")
                    if not event_source:
                        rejection_reasons.add("missing_source_type")
                    elif event_source.lower() == "captured_passthrough":
                        rejection_reasons.add("captured_source_type")
                    event_metadata = _metadata(first_row["event_metadata"])
                    event_scope, scope_reason = _scope_identity(
                        metadata=event_metadata,
                        column_scope=first_row["event_scope"],
                        column_scope_id=first_row["event_scope_id"],
                        column_profile=first_row["event_profile"],
                        require_scope=True,
                    )
                    if scope_reason is not None:
                        rejection_reasons.add(scope_reason)

            facts: dict[str, _FactObservation] = {}
            for fact_id in fact_ids:
                fact_rows = [row for row in group_rows if str(row["fact_id"]) == fact_id]
                first = fact_rows[0]
                reasons: set[str] = set()
                content = first["fact_content"]
                scope: _ScopeIdentity | None = None
                raw_version = first["fact_version"]
                version = raw_version if isinstance(raw_version, int) and raw_version > 0 else 0
                valid_evidence_ids: list[str] = []
                if content is None:
                    reasons.add("missing_fact")
                    content = ""
                else:
                    if version == 0:
                        reasons.add("invalid_fact_version")
                    fact_metadata = _metadata(first["fact_metadata"])
                    scope, scope_reason = _scope_identity(
                        metadata=fact_metadata,
                        column_profile=first["fact_profile"],
                        require_scope=True,
                    )
                    if scope_reason is not None:
                        reasons.add(scope_reason)
                    if fact_id in suppressed_fact_ids:
                        reasons.add("suppressed_fact")
                    if (fact_metadata or {}).get("source") == "captured_passthrough":
                        reasons.add("captured_evidence")
                    if (fact_metadata or {}).get("assertion_status") != "asserted":
                        reasons.add("nonasserted_fact")
                    for evidence_row in fact_rows:
                        if _normalize_text(str(evidence_row["excerpt"])) == _normalize_text(
                            str(content)
                        ):
                            valid_evidence_ids.append(str(evidence_row["evidence_id"]))
                    if not valid_evidence_ids:
                        reasons.add("unverifiable_current_content")
                facts[fact_id] = _FactObservation(
                    fact_id=fact_id,
                    content=str(content),
                    version=version,
                    scope=scope,
                    evidence_ids=sorted(valid_evidence_ids),
                    reasons=reasons,
                )
                rejection_reasons.update(reasons)

            if len(fact_ids) > fanout_limit:
                rejection_reasons.add("fanout_limit_exceeded")
                status = "overflow"
            elif rejection_reasons & {
                "missing_source_event_id",
                "missing_source_event",
                "suppressed_source_event",
                "unsupported_event_role",
                "missing_source_type",
                "captured_source_type",
                "malformed_scope_metadata",
                "profile_metadata_mismatch",
                "scope_metadata_mismatch",
                "scope_id_metadata_mismatch",
                "missing_scope",
                "invalid_scope",
                "missing_scope_id",
            }:
                status = "rejected"
            else:
                status = "pending"

            eligible = [
                fact
                for fact in facts.values()
                if not fact.reasons and fact.scope is not None and event_scope is not None
            ]
            if status == "pending":
                assert event_scope is not None
                for first_fact, second_fact in itertools.combinations(eligible, 2):
                    assert first_fact.scope is not None and second_fact.scope is not None
                    if (
                        not _scopes_can_connect(first_fact.scope, second_fact.scope)
                        or not _scopes_can_connect(event_scope, first_fact.scope)
                        or not _scopes_can_connect(event_scope, second_fact.scope)
                    ):
                        rejection_reasons.add("scope_mismatch")
                        continue
                    evidence_ids = sorted(set(first_fact.evidence_ids + second_fact.evidence_ids))
                    relations.append(
                        {
                            "relation_id": _relation_id(
                                source_event_id or "", first_fact.fact_id, second_fact.fact_id
                            ),
                            "source_event_id": source_event_id,
                            "fact_ids": [first_fact.fact_id, second_fact.fact_id],
                            "fact_versions": {
                                first_fact.fact_id: first_fact.version,
                                second_fact.fact_id: second_fact.version,
                            },
                            "evidence_ids": evidence_ids,
                            "scope": first_fact.scope.as_json(),
                            "fact_scopes": {
                                first_fact.fact_id: first_fact.scope.as_json(),
                                second_fact.fact_id: second_fact.scope.as_json(),
                            },
                            "source_scope": event_scope.as_json(),
                            "source_role": event_role,
                            "source_type": event_source,
                        }
                    )
            if relations and any(item["source_event_id"] == source_event_id for item in relations):
                status = "eligible" if status == "pending" else status
            elif status == "pending":
                status = "insufficient_facts" if len(eligible) < 2 else "rejected"
            if status == "rejected" and not rejection_reasons:
                rejection_reasons.add("no_scope_compatible_pair")
            groups.append(
                {
                    "source_event_id": source_event_id,
                    "status": status,
                    "fact_ids": fact_ids,
                    "eligible_fact_ids": sorted(fact.fact_id for fact in eligible),
                    "fact_count": len(fact_ids),
                    "fact_versions": {fact.fact_id: fact.version for fact in facts.values()},
                    "evidence_ids": sorted(str(row["evidence_id"]) for row in group_rows),
                    "fact_rejections": {
                        fact.fact_id: sorted(fact.reasons)
                        for fact in facts.values()
                        if fact.reasons
                    },
                    "source_scope": event_scope.as_json() if event_scope is not None else None,
                    "source_role": event_role,
                    "source_type": event_source,
                    "rejection_reasons": sorted(rejection_reasons),
                }
            )

        relations.sort(key=lambda item: item["relation_id"])
        groups.sort(
            key=lambda item: "" if item["source_event_id"] is None else item["source_event_id"]
        )
        summary = {
            "source_event_groups": len(groups),
            "eligible_source_event_groups": sum(group["status"] == "eligible" for group in groups),
            "relation_count": len(relations),
            "rejected_source_event_groups": sum(group["status"] == "rejected" for group in groups),
            "insufficient_source_event_groups": sum(
                group["status"] == "insufficient_facts" for group in groups
            ),
            "overflow_source_event_groups": sum(group["status"] == "overflow" for group in groups),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "read_only_snapshot": True,
            "fanout_limit": fanout_limit,
            "relation_basis": "fact_evidence.source_event_id + retained episodic_events",
            "groups": groups,
            "relations": relations,
            "summary": summary,
        }
    finally:
        connection.rollback()
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--fanout-limit", type=int, default=16)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            inspect_source_associations(args.db_path, fanout_limit=args.fanout_limit), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
