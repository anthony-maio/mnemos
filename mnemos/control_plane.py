"""
mnemos/control_plane.py — Local onboarding/control-plane service.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal, Mapping

from .engine import MnemosEngine
from .health import run_health_checks
from .hosts import apply_host_integration, preview_host_integration
from .inspectability import build_chunk_inspection
from .runtime import (
    build_embedder_from_settings,
    build_llm_from_settings,
    build_mnemos_config_from_settings,
    build_store_from_settings,
)
from .settings import AppSettings, import_existing_setup, load_settings, save_settings
from .types import Interaction

HOST_SUPPORT: dict[str, dict[str, Any]] = {
    "claude-code": {
        "name": "claude-code",
        "label": "Claude Code",
        "tier": "Tier 1",
        "status": "Supported",
        "recommended": True,
        "summary": "Primary Mnemos path with the strongest daily-use validation.",
        "best_path": "Apply the Claude Code integration, then run the smoke check before daily use.",
    },
    "cursor": {
        "name": "cursor",
        "label": "Cursor",
        "tier": "Tier 2",
        "status": "Documented",
        "recommended": False,
        "summary": "Usable through MCP plus repo rules, but not promoted to parity in 0.7.0.",
        "best_path": "Apply the Cursor integration and pair it with the repo rule pack.",
    },
    "codex": {
        "name": "codex",
        "label": "Codex",
        "tier": "Tier 2",
        "status": "Documented",
        "recommended": False,
        "summary": "Supported through MCP plus AGENTS.md guidance; still a documented path, not Tier 1.",
        "best_path": "Apply the Codex MCP config, then keep the AGENTS.md recall/store loop in place.",
    },
}

SETUP_PRESETS: list[dict[str, Any]] = [
    {
        "id": "starter-sqlite",
        "label": "Starter SQLite",
        "recommended": False,
        "description": "Default shipped path: one SQLite file, guided setup, and minimal moving parts.",
        "payload": {
            "storage": {"type": "sqlite"},
            "sleep": {
                "recall_gated_plasticity_enabled": False,
                "recall_min_supporting_episodes": 2,
                "recall_similarity_threshold": 0.82,
            },
        },
    },
    {
        "id": "stable-memory",
        "label": "Stable Memory",
        "recommended": True,
        "description": "Recommended for noisy or longer-running sessions. Enables recall-gated consolidation.",
        "payload": {
            "storage": {"type": "sqlite"},
            "sleep": {
                "recall_gated_plasticity_enabled": True,
                "recall_min_supporting_episodes": 2,
                "recall_similarity_threshold": 0.82,
            },
        },
    },
]


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _mask_secret_fields(payload: dict[str, Any]) -> dict[str, Any]:
    masked = dict(payload)
    providers = masked.get("providers")
    if not isinstance(providers, dict):
        return masked
    safe_providers: dict[str, Any] = {}
    for name, values in providers.items():
        if not isinstance(values, dict):
            safe_providers[name] = values
            continue
        safe_values = dict(values)
        if safe_values.get("api_key"):
            safe_values["api_key"] = ""
            safe_values["configured"] = True
        else:
            safe_values["configured"] = False
        if safe_values.get("password"):
            safe_values["password"] = ""
            safe_values["configured"] = True
        safe_providers[name] = safe_values
    masked["providers"] = safe_providers
    return masked


def _host_catalog() -> list[dict[str, Any]]:
    return [dict(value) for value in HOST_SUPPORT.values()]


def _host_entry(name: str) -> dict[str, Any]:
    return dict(HOST_SUPPORT.get(name, HOST_SUPPORT["claude-code"]))


def _default_activity(status: str = "not_run") -> dict[str, Any]:
    return {"present": False, "status": status}


class ControlPlaneService:
    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        home: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        global_config_path: str | Path | None = None,
    ) -> None:
        self.cwd = Path.cwd() if cwd is None else Path(cwd)
        self.home = Path.home() if home is None else Path(home)
        self.env = dict(env or {})
        self.global_config_path = None if global_config_path is None else Path(global_config_path)
        self._last_smoke_report: dict[str, Any] | None = None
        self._last_consolidation_report: dict[str, Any] | None = None

    def _config_env(self) -> dict[str, str]:
        merged = dict(self.env)
        if self.global_config_path is not None:
            merged["MNEMOS_CONFIG_PATH"] = str(self.global_config_path)
        return merged

    def _resolved_settings(self) -> Any:
        return load_settings(
            env=self._config_env(),
            cwd=self.cwd,
            global_config_path=self.global_config_path,
            default_store_type="sqlite",
        )

    def _setup_story(self) -> dict[str, Any]:
        return {
            "primary_store": "sqlite",
            "advanced_store_types": ["memory"],
            "recommended_host": "claude-code",
            "advanced_fields": ["providers.*.base_url", "providers.*.api_key"],
        }

    def _settings_payload(self, settings: AppSettings) -> dict[str, Any]:
        return _mask_secret_fields(settings.model_dump(mode="python", exclude_none=True))

    def get_settings_view(self) -> dict[str, Any]:
        resolved = self._resolved_settings()
        preferred_host = resolved.settings.onboarding.preferred_host
        return {
            "settings": self._settings_payload(resolved.settings),
            "paths": {
                "global_config": str(resolved.global_config_path),
                "project_config": (
                    None
                    if resolved.project_config_path is None
                    else str(resolved.project_config_path)
                ),
            },
            "warnings": resolved.warnings,
            "presets": [dict(preset) for preset in SETUP_PRESETS],
            "setup_story": self._setup_story(),
            "host_support": {
                "recommended": "claude-code",
                "current": _host_entry(preferred_host),
                "hosts": _host_catalog(),
            },
        }

    def save_settings(
        self,
        payload: Mapping[str, Any],
        *,
        scope: Literal["global", "project"],
    ) -> dict[str, Any]:
        resolved = self._resolved_settings()
        current = resolved.settings.model_dump(mode="python", exclude_none=False)
        merged = _deep_merge(current, payload)

        providers = merged.get("providers")
        current_providers = current.get("providers")
        if isinstance(providers, dict) and isinstance(current_providers, dict):
            for provider_name, provider_values in providers.items():
                if not isinstance(provider_values, dict):
                    continue
                current_values = current_providers.get(provider_name)
                if not isinstance(current_values, dict):
                    continue
                api_key = provider_values.get("api_key")
                if api_key == "":
                    provider_values["api_key"] = current_values.get("api_key")

        settings = AppSettings.model_validate(merged)
        if scope == "global":
            target = resolved.global_config_path
        else:
            target = self.cwd / ".mnemos" / "mnemos.toml"
        save_settings(settings, target, scope=scope)
        return {"saved": True, "scope": scope, "path": str(target)}

    def import_existing_setup(self) -> dict[str, Any]:
        imported = import_existing_setup(
            env=self.env,
            cwd=self.cwd,
            home=self.home,
            global_config_path=self.global_config_path,
        )
        return {
            "settings": self._settings_payload(imported.settings),
            "sources": imported.sources,
            "paths": {
                "global_config": str(
                    self.global_config_path
                    if self.global_config_path is not None
                    else load_settings(
                        env=self._config_env(),
                        cwd=self.cwd,
                        default_store_type="sqlite",
                    ).global_config_path
                ),
                "project_config": str(self.cwd / ".mnemos" / "mnemos.toml"),
            },
            "warnings": [],
            "presets": [dict(preset) for preset in SETUP_PRESETS],
            "setup_story": self._setup_story(),
            "host_support": {
                "recommended": "claude-code",
                "current": _host_entry(imported.settings.onboarding.preferred_host),
                "hosts": _host_catalog(),
            },
        }

    def preview_integration(
        self, host: Literal["claude-code", "cursor", "codex"]
    ) -> dict[str, Any]:
        resolved = self._resolved_settings()
        preview = preview_host_integration(
            host,
            mnemos_config_path=resolved.global_config_path,
            cwd=self.cwd,
            home=self.home,
        )
        return {
            "host": host,
            "config_path": str(preview.config_path),
            "preview": preview.preview_text,
            "support": _host_entry(host),
        }

    def apply_integration(self, host: Literal["claude-code", "cursor", "codex"]) -> dict[str, Any]:
        resolved = self._resolved_settings()
        result = apply_host_integration(
            host,
            mnemos_config_path=resolved.global_config_path,
            cwd=self.cwd,
            home=self.home,
        )
        return {
            "host": host,
            "config_path": str(result.config_path),
            "backup_path": None if result.backup_path is None else str(result.backup_path),
            "preview": result.preview_text,
            "support": _host_entry(host),
        }

    def health_report(self) -> dict[str, Any]:
        return run_health_checks(env=self._config_env(), default_store_type="sqlite")

    def _build_engine(self, settings: AppSettings) -> MnemosEngine:
        return MnemosEngine(
            config=build_mnemos_config_from_settings(settings),
            llm=build_llm_from_settings(settings),
            embedder=build_embedder_from_settings(settings),
            store=build_store_from_settings(settings),
        )

    def get_memory_snapshot(self, *, limit: int = 10) -> dict[str, Any]:
        resolved = self._resolved_settings()
        store = build_store_from_settings(resolved.settings)
        try:
            chunks = sorted(store.get_all(), key=lambda chunk: chunk.updated_at, reverse=True)
            recent = chunks[:limit]
            counts_by_scope = {"global": 0, "project": 0, "workspace": 0}
            consolidated_chunks: list[Any] = []
            recall_supported_chunks: list[Any] = []

            for chunk in chunks:
                scope = str(chunk.metadata.get("scope", "global"))
                if scope in counts_by_scope:
                    counts_by_scope[scope] += 1
                if chunk.metadata.get("source") == "sleep_consolidation":
                    consolidated_chunks.append(chunk)
                recall_support_value = chunk.metadata.get("recall_supporting_episodes")
                if isinstance(recall_support_value, int):
                    recall_supported_chunks.append(chunk)

            latest_consolidation_time = max(
                (
                    str(chunk.metadata.get("consolidation_time"))
                    for chunk in consolidated_chunks
                    if chunk.metadata.get("consolidation_time")
                ),
                default=None,
            )
            latest_batch = [
                chunk
                for chunk in consolidated_chunks
                if chunk.metadata.get("consolidation_time") == latest_consolidation_time
            ]

            max_supporting_episodes = max(
                (
                    int(chunk.metadata.get("recall_supporting_episodes", 0))
                    for chunk in recall_supported_chunks
                ),
                default=0,
            )
            max_support_score = max(
                (
                    float(chunk.metadata.get("recall_support_score", 0.0))
                    for chunk in recall_supported_chunks
                ),
                default=0.0,
            )

            return {
                "count": len(chunks),
                "recent_count": len(recent),
                "counts_by_scope": counts_by_scope,
                "recent": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "scope": chunk.metadata.get("scope", "global"),
                        "scope_id": chunk.metadata.get("scope_id"),
                        "updated_at": chunk.updated_at.isoformat(),
                        "access_count": chunk.access_count,
                        "stored_by": chunk.metadata.get("source"),
                    }
                    for chunk in recent
                ],
                "last_consolidation": {
                    "present": bool(latest_batch),
                    "at": latest_consolidation_time,
                    "source": "sleep_consolidation" if latest_batch else None,
                    "stored_chunk_count": len(latest_batch),
                    "facts_preview": [chunk.content for chunk in latest_batch[:3]],
                },
                "recall_gated": {
                    "enabled": bool(
                        resolved.settings.sleep.recall_gated_plasticity_enabled
                        or recall_supported_chunks
                    ),
                    "stored_chunk_count": len(recall_supported_chunks),
                    "max_supporting_episodes": max_supporting_episodes,
                    "max_support_score": round(max_support_score, 4),
                },
            }
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def get_memory_detail(
        self,
        chunk_id: str,
        *,
        query: str | None = None,
        current_scope: str = "project",
        scope_id: str | None = None,
        allowed_scopes: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolved_settings()
        engine = self._build_engine(resolved.settings)
        try:
            payload = build_chunk_inspection(
                engine,
                chunk_id,
                query=query,
                current_scope=current_scope,
                scope_id=scope_id,
                allowed_scopes=allowed_scopes,
            )
            if payload is None:
                raise KeyError(chunk_id)
            return payload
        finally:
            close = getattr(engine.store, "close", None)
            if callable(close):
                close()

    def _build_consolidation_activity(
        self,
        *,
        result: Any,
        before_stats: Mapping[str, Any],
        after_stats: Mapping[str, Any],
        triggered_by: str,
    ) -> dict[str, Any]:
        recall_supported = int(after_stats.get("total_recall_supported_facts", 0)) - int(
            before_stats.get("total_recall_supported_facts", 0)
        )
        recall_filtered = int(after_stats.get("total_recall_filtered_facts", 0)) - int(
            before_stats.get("total_recall_filtered_facts", 0)
        )
        return {
            "present": True,
            "status": "pass",
            "triggered_by": triggered_by,
            "facts_extracted": len(result.facts_extracted),
            "facts_preview": list(result.facts_extracted[:3]),
            "chunks_pruned": result.chunks_pruned,
            "duration_seconds": round(result.duration_seconds, 3),
            "recall_supported": max(recall_supported, 0),
            "recall_filtered": max(recall_filtered, 0),
        }

    def run_consolidation(self) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            resolved = self._resolved_settings()
            engine = self._build_engine(resolved.settings)
            try:
                before_stats = engine.sleep_daemon.get_stats()
                result = await engine.consolidate()
                after_stats = engine.sleep_daemon.get_stats()
                report = self._build_consolidation_activity(
                    result=result,
                    before_stats=before_stats,
                    after_stats=after_stats,
                    triggered_by="manual",
                )
                self._last_consolidation_report = report
                return report
            finally:
                close = getattr(engine.store, "close", None)
                if callable(close):
                    close()

        try:
            return asyncio.run(_run())
        except Exception as exc:
            report = {
                "present": True,
                "status": "fail",
                "triggered_by": "manual",
                "error": str(exc),
            }
            self._last_consolidation_report = report
            return report

    def run_smoke_tests(self) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            resolved = self._resolved_settings()
            engine = self._build_engine(resolved.settings)
            steps: dict[str, str] = {}
            try:
                await engine.process(
                    Interaction(role="user", content="Smoke test preference: use concise replies."),
                    scope="project",
                    scope_id="mnemos-control-plane",
                )
                steps["store"] = "pass"

                retrieved = await engine.retrieve(
                    "concise replies",
                    top_k=3,
                    reconsolidate=False,
                    current_scope="project",
                    scope_id="mnemos-control-plane",
                    allowed_scopes=("project", "global"),
                )
                steps["retrieve"] = "pass" if retrieved else "warn"

                before_stats = engine.sleep_daemon.get_stats()
                result = await engine.consolidate()
                after_stats = engine.sleep_daemon.get_stats()
                steps["consolidate"] = "pass"
                self._last_consolidation_report = self._build_consolidation_activity(
                    result=result,
                    before_stats=before_stats,
                    after_stats=after_stats,
                    triggered_by="smoke",
                )

                health = self.health_report()
                steps["health"] = "pass" if health["status"] == "ready" else "warn"
            finally:
                close = getattr(engine.store, "close", None)
                if callable(close):
                    close()

            status = "fail" if any(step == "fail" for step in steps.values()) else "pass"
            report = {"status": status, "steps": steps}
            self._last_smoke_report = {"present": True, **report}
            return report

        try:
            return asyncio.run(_run())
        except Exception as exc:
            report = {
                "status": "fail",
                "steps": {
                    "store": "fail",
                    "retrieve": "fail",
                    "consolidate": "fail",
                    "health": "fail",
                },
                "error": str(exc),
            }
            self._last_smoke_report = {"present": True, **report}
            return report

    def get_dashboard_view(self) -> dict[str, Any]:
        resolved = self._resolved_settings()
        health = self.health_report()
        memory = self.get_memory_snapshot(limit=8)
        current_host = resolved.settings.onboarding.preferred_host
        next_actions = list(health.get("next_actions", []))
        current_host_entry = _host_entry(current_host)
        if current_host_entry["tier"] != "Tier 1":
            next_actions.append(
                f"{current_host_entry['label']} remains a documented path in 0.7.0. Use Claude Code for the strongest validated workflow."
            )
        if self._last_smoke_report is None:
            next_actions.append(
                "Run the smoke check once after applying a host integration to verify the setup end to end."
            )

        deduped_actions: list[str] = []
        for action in next_actions:
            if action not in deduped_actions:
                deduped_actions.append(action)

        return {
            "setup": {
                "status": health["status"],
                "store_type": resolved.settings.storage.type,
                "sqlite_path": resolved.settings.storage.sqlite_path,
                "global_config": str(resolved.global_config_path),
                "project_config": (
                    None
                    if resolved.project_config_path is None
                    else str(resolved.project_config_path)
                ),
            },
            "host": {
                "recommended": "claude-code",
                "current": current_host_entry,
                "hosts": _host_catalog(),
            },
            "memory": memory,
            "health": health,
            "activity": {
                "last_smoke_test": self._last_smoke_report or _default_activity(),
                "last_consolidation": self._last_consolidation_report or _default_activity(),
            },
            "next_actions": deduped_actions,
        }
