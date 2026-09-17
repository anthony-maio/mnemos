"""
mnemos/control_plane.py — Local onboarding/control-plane service.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import threading
from pathlib import Path
from typing import Any, Literal, Mapping

from .client import MnemosClient
from .health import run_health_checks
from .hosts import apply_host_integration, preview_host_integration
from .routing_review import ack_routing_failure, list_routing_failures
from .scopes import ScopeContext, ScopeRegistry, default_registry_path
from .settings import AppSettings, import_existing_setup, load_settings, save_settings

HOST_SUPPORT: dict[str, dict[str, Any]] = {
    "claude-code": {
        "name": "claude-code",
        "label": "Claude Code",
        "tier": "Tier 2",
        "status": "Supported",
        "recommended": True,
        "summary": "MCP and adapter contracts are tested; installed-host continuity is pending.",
        "best_path": "Apply project MCP configuration, install the Mnemos Claude plugin for automatic capture, and validate a real host session.",
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
        "summary": "Native hooks and MCP are implemented; CLI and desktop validation remain separate gates.",
        "best_path": "Preview the native hooks and MCP configuration for your enabled project.",
    },
    "hermes": {
        "name": "hermes",
        "label": "Hermes",
        "tier": "Tier 2",
        "status": "Documented",
        "recommended": False,
        "summary": "Native MemoryProvider support is available when the installed Hermes version exposes it.",
        "best_path": "Configure the Hermes provider descriptor, then use manual MCP fallback when native support is unavailable.",
    },
}

SETUP_PRESETS: list[dict[str, Any]] = [
    {
        "id": "starter-sqlite",
        "label": "Starter SQLite",
        "recommended": True,
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
        "label": "Embedded API recall experiment",
        "recommended": False,
        "description": "Legacy embedded API setting. Does not change managed-service extraction or establish a quality improvement.",
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
        client: Any | None = None,
    ) -> None:
        self.cwd = Path.cwd() if cwd is None else Path(cwd)
        self.home = Path.home() if home is None else Path(home)
        self.env = dict(env or {})
        self.global_config_path = None if global_config_path is None else Path(global_config_path)
        self._last_smoke_report: dict[str, Any] | None = None
        self._last_consolidation_report: dict[str, Any] | None = None
        self.client = client or MnemosClient()
        self._scope_registry: ScopeRegistry | None = None

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
        return {
            "saved": True,
            "scope": scope,
            "path": str(target),
            "restart_required": True,
            "runtime_note": "The owning service reads global configuration at startup. Project provider overrides do not change it.",
        }

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
        self, host: Literal["claude-code", "cursor", "codex", "hermes"]
    ) -> dict[str, Any]:
        resolved = self._resolved_settings()
        preview = preview_host_integration(
            host,
            mnemos_config_path=resolved.global_config_path,
            cwd=self.cwd,
            home=self.home,
            integration_env=self._integration_env(),
        )
        return {
            "host": host,
            "config_path": str(preview.config_path),
            "preview": preview.preview_text,
            "support": _host_entry(host),
        }

    def apply_integration(
        self, host: Literal["claude-code", "cursor", "codex", "hermes"]
    ) -> dict[str, Any]:
        resolved = self._resolved_settings()
        result = apply_host_integration(
            host,
            mnemos_config_path=resolved.global_config_path,
            cwd=self.cwd,
            home=self.home,
            integration_env=self._integration_env(),
        )
        return {
            "host": host,
            "config_path": str(result.config_path),
            "backup_path": None if result.backup_path is None else str(result.backup_path),
            "preview": result.preview_text,
            "support": _host_entry(host),
        }

    def health_report(self) -> dict[str, Any]:
        report = run_health_checks(env=self._config_env(), default_store_type="sqlite")
        configured = report.get("provider_state", {})
        observed = report.get("service", {}).get("provider_state", {})
        report["provider_readiness"] = {
            "configured": configured,
            "observed": observed,
        }
        try:
            service_health = self._service_request("health", {})
            report["service"] = service_health
            report["provider_readiness"]["observed"] = service_health.get(
                "provider_state", {"cognition": {"status": "not_observed"}}
            )
            cognition = service_health.get("provider_state", {}).get("cognition", {})
            report["status"] = (
                "ready"
                if service_health.get("status") == "running"
                and cognition.get("input_specific_extraction") is True
                else "degraded"
            )
        except Exception as exc:
            report["status"] = "degraded"
            report["service"] = {
                "status": "not_observed",
                "error": str(exc),
            }
        return report

    def _run_sync(self, awaitable: Any) -> Any:
        """Run one bounded client call from both sync handlers and async callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        result: list[Any] = []
        failure: list[BaseException] = []

        def runner() -> None:
            try:
                result.append(asyncio.run(awaitable))
            except BaseException as exc:  # pragma: no cover - exercised by async callers
                failure.append(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=15.0)
        if thread.is_alive():
            raise TimeoutError("Mnemos service request exceeded the UI timeout")
        if failure:
            raise failure[0]
        return result[0] if result else None

    def _service_request(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = getattr(self.client, "request", None)
        if not callable(request):
            raise TypeError("Configured Mnemos client does not expose request()")
        result = request(operation, dict(payload), queue_on_failure=False)
        if inspect.isawaitable(result):
            result = self._run_sync(result)
        if not isinstance(result, dict):
            raise TypeError("Mnemos service returned a non-object response")
        return dict(result)

    def _registry_path(self) -> Path:
        configured = self.env.get("MNEMOS_SCOPE_REGISTRY") or os.environ.get(
            "MNEMOS_SCOPE_REGISTRY"
        )
        return Path(configured).expanduser() if configured else default_registry_path()

    def _configured_profile(self) -> str | None:
        selected = (
            self.env.get("MNEMOS_USER_PROFILE_ID")
            or os.environ.get("MNEMOS_USER_PROFILE_ID")
            or self._resolved_settings().settings.onboarding.user_profile_id
        )
        return selected.strip() if selected and selected.strip() else None

    def _integration_env(self) -> dict[str, str]:
        values = {"MNEMOS_SCOPE_REGISTRY": str(self._registry_path())}
        profile = self._configured_profile()
        if profile:
            values["MNEMOS_USER_PROFILE_ID"] = profile
        descriptor = self.env.get("MNEMOS_RUNTIME_DESCRIPTOR") or os.environ.get(
            "MNEMOS_RUNTIME_DESCRIPTOR"
        )
        if descriptor:
            values["MNEMOS_RUNTIME_DESCRIPTOR"] = descriptor
        return values

    def _context_payload(self) -> dict[str, Any]:
        profile = self._configured_profile()
        if not profile:
            raise ValueError("MNEMOS_USER_PROFILE_ID is required for the Mnemos UI")
        if self._scope_registry is None:
            self._scope_registry = ScopeRegistry.load(self._registry_path())
        session = self.env.get("MNEMOS_UI_SESSION_ID") or os.environ.get("MNEMOS_UI_SESSION_ID")
        if not session:
            digest = hashlib.sha256(str(self.cwd.resolve()).encode("utf-8")).hexdigest()[:16]
            session = f"ui-{digest}"
        context = self._scope_registry.resolve(
            self.cwd, user_profile_id=profile, session_id=session, host="manual"
        )
        return context.model_dump(mode="json")

    def get_scope_status(self) -> dict[str, Any]:
        profile = self._configured_profile()
        payload: dict[str, Any] = {
            "configured": bool(profile),
            "profile": profile,
            "registry_path": str(self._registry_path()),
            "project": {"enabled": False, "id": None},
        }
        if not profile:
            payload["status"] = "not_configured"
            payload["error"] = "MNEMOS_USER_PROFILE_ID is required for the Mnemos UI"
            return payload
        try:
            context = self._context_payload()
            payload["status"] = "ready"
            payload["project"] = {"enabled": True, "id": context.get("project_id")}
            payload["workspace_id"] = context.get("workspace_id")
        except Exception as exc:
            payload["status"] = "not_ready"
            payload["error"] = str(exc)
        return payload

    def configure_project(self, profile: str, enabled: bool) -> dict[str, Any]:
        result = self._service_request(
            "configure_project",
            {"path": str(self.cwd.resolve()), "user_profile_id": profile, "enabled": enabled},
        )
        if result.get("status") == "committed":
            self.env["MNEMOS_USER_PROFILE_ID"] = profile.strip()
            self._scope_registry = None
            resolved = self._resolved_settings()
            # Persist the user's selection without copying project provider
            # overrides into the owning service's global configuration.
            global_settings = load_settings(
                env=self._config_env(),
                cwd=resolved.global_config_path.parent,
                global_config_path=resolved.global_config_path,
                default_store_type="sqlite",
            ).settings
            global_settings.onboarding.user_profile_id = profile.strip()
            save_settings(global_settings, resolved.global_config_path, scope="global")
        return result

    def get_jobs(self, *, status: str | None = "exhausted", limit: int = 50) -> dict[str, Any]:
        return self._service_request(
            "jobs", {"context": self._context_payload(), "status": status, "limit": limit}
        )

    def retry_job(self, job_id: str) -> dict[str, Any]:
        return self._service_request(
            "retry_job", {"context": self._context_payload(), "job_id": job_id}
        )

    def _routing_review_directory(self) -> str | None:
        return self.env.get("MNEMOS_ROUTING_REVIEW_DIR") or os.environ.get(
            "MNEMOS_ROUTING_REVIEW_DIR"
        )

    def _routing_review_profile(self, *, unconfigured: bool) -> str | None:
        if unconfigured:
            return None
        profile = self._configured_profile()
        if not profile:
            raise ValueError("MNEMOS_USER_PROFILE_ID is required to review routing failures")
        return profile

    def get_routing_review(self, *, unconfigured: bool = False, limit: int = 256) -> dict[str, Any]:
        profile = self._routing_review_profile(unconfigured=unconfigured)
        records = list_routing_failures(
            configured_profile=profile,
            directory=self._routing_review_directory(),
            limit=limit,
        )
        return {
            "schema_version": 1,
            "status": "committed",
            "read_only": True,
            "unconfigured": unconfigured,
            "records": records,
        }

    def ack_routing_review(self, record_id: str, *, unconfigured: bool = False) -> dict[str, Any]:
        profile = self._routing_review_profile(unconfigured=unconfigured)
        acknowledged = ack_routing_failure(
            record_id,
            configured_profile=profile,
            directory=self._routing_review_directory(),
        )
        return {
            "schema_version": 1,
            "status": "committed",
            "acknowledged": acknowledged,
            "record_id": record_id,
            "unconfigured": unconfigured,
        }

    def get_legacy_review(self, *, limit: int = 50, after_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"context": self._context_payload(), "limit": limit}
        if after_id:
            payload["after_id"] = after_id
        return self._service_request("legacy_review", payload)

    def get_memory_snapshot(self, *, limit: int = 10) -> dict[str, Any]:
        try:
            response = self._service_request(
                "list",
                {
                    "context": self._context_payload(),
                    "allowed_scopes": ["project", "workspace", "global"],
                    "limit": 500,
                },
            )
        except Exception as exc:
            return {
                "status": "not_configured",
                "error": str(exc),
                "count": 0,
                "recent_count": 0,
                "recent": [],
                "counts_by_scope": {"global": 0, "project": 0, "workspace": 0},
                "retrieval_profiles": {"default": "baseline", "experimental": ["biomimetic"]},
                "jobs": {"status": "not_observed"},
            }
        facts = response.get("facts", [])
        if not isinstance(facts, list):
            facts = []
        counts = {"global": 0, "project": 0, "workspace": 0}
        recent: list[dict[str, Any]] = []
        consolidated: list[dict[str, Any]] = []
        supported: list[dict[str, Any]] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            scope = str(fact.get("metadata", {}).get("scope", fact.get("scope", "global")))
            if scope in counts:
                counts[scope] += 1
            metadata = fact.get("metadata", {}) if isinstance(fact.get("metadata"), dict) else {}
            item = {
                "id": fact.get("id"),
                "content": fact.get("content", ""),
                "scope": scope,
                "scope_id": metadata.get("scope_id", fact.get("scope_id")),
                "updated_at": fact.get("updated_at"),
                "access_count": fact.get("access_count", 0),
                "stored_by": metadata.get("source"),
                "version": fact.get("version", 1),
                "sources": fact.get("sources", []),
                "metadata": metadata,
            }
            recent.append(item)
            if metadata.get("source") == "sleep_consolidation":
                consolidated.append(item)
            if isinstance(metadata.get("recall_supporting_episodes"), int):
                supported.append(metadata)
        consolidation_times = [
            str((item.get("metadata") or {}).get("consolidation_time"))
            for item in consolidated
            if (item.get("metadata") or {}).get("consolidation_time")
        ]
        latest = max(consolidation_times, default=None)
        latest_batch = [
            item
            for item in consolidated
            if str((item.get("metadata") or {}).get("consolidation_time")) == latest
        ]
        resolved = self._resolved_settings()
        jobs = response.get("jobs")
        if not isinstance(jobs, dict):
            jobs = {"status": "not_observed"}
        return {
            "status": response.get("status", "ready"),
            "count": len(facts),
            "count_at_least": len(facts) >= 500,
            "listing_note": "Preview of up to 500 facts by ID. Use mnemos list for the complete scoped listing.",
            "recent_count": min(len(recent), max(limit, 0)),
            "counts_by_scope": counts,
            "recent": recent[: max(limit, 0)],
            "last_consolidation": {
                "present": bool(latest_batch),
                "at": latest,
                "source": "sleep_consolidation" if latest_batch else None,
                "stored_chunk_count": len(latest_batch),
                "facts_preview": [item["content"] for item in latest_batch[:3]],
            },
            "recall_gated": {
                "enabled": bool(
                    resolved.settings.sleep.recall_gated_plasticity_enabled or supported
                ),
                "stored_chunk_count": len(supported),
                "max_supporting_episodes": max(
                    (int(item.get("recall_supporting_episodes", 0)) for item in supported),
                    default=0,
                ),
                "max_support_score": round(
                    max(
                        (float(item.get("recall_support_score", 0.0)) for item in supported),
                        default=0.0,
                    ),
                    4,
                ),
            },
            "retrieval_profiles": {"default": "baseline", "experimental": ["biomimetic"]},
            "jobs": jobs,
        }

    def get_memory_detail(
        self,
        chunk_id: str,
        *,
        query: str | None = None,
        current_scope: str = "project",
        scope_id: str | None = None,
        allowed_scopes: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        context = self._context_payload()
        inspected = self._service_request(
            "inspect",
            {"context": context, "fact_id": chunk_id},
        )
        fact = inspected.get("fact")
        if not isinstance(fact, dict):
            raise KeyError(chunk_id)
        metadata = fact.get("metadata", {}) if isinstance(fact.get("metadata"), dict) else {}
        result: dict[str, Any] = {
            "id": fact.get("id", chunk_id),
            "content": fact.get("content", ""),
            "scope": metadata.get("scope", "unassigned"),
            "scope_id": metadata.get("scope_id"),
            "salience": fact.get("salience", 0.5),
            "version": fact.get("version", 1),
            "access_count": fact.get("access_count", 0),
            "created_at": fact.get("created_at"),
            "updated_at": fact.get("updated_at"),
            "cognitive_state": fact.get("cognitive_state"),
            "provenance": {
                "stored_by": metadata.get("source", "unknown"),
                "ingest_channel": metadata.get("ingest_channel"),
                "encoding_reason": metadata.get("encoding_reason"),
                "original_timestamp": metadata.get("original_timestamp"),
                "reconsolidated_at": metadata.get("reconsolidated_at"),
            },
            "history": inspected.get("revisions", []),
            "evidence": inspected.get("evidence", []),
            "original_source_status": inspected.get("original_source_status", "no_source_evidence"),
            "graph": {"present": False, "neighbor_count": 0, "neighbors": []},
            "retrieval": None,
            "metadata": metadata,
        }
        if query and query.strip():
            retrieval = self._service_request(
                "retrieve",
                {
                    "query": query.strip(),
                    "context": context,
                    "allowed_scopes": list(allowed_scopes or ("project", "workspace", "global")),
                    "top_k": 10,
                    "max_context_tokens": 1200,
                    "profile": "baseline",
                },
            )
            candidate = next(
                (item for item in retrieval.get("facts", []) if item.get("id") == chunk_id), None
            )
            if candidate is None:
                result["retrieval"] = {
                    "query": query.strip(),
                    "status": "not_observed",
                    "semantic_rank": None,
                    "explanation": [
                        "No matching retrieval trace is available for this fact version."
                    ],
                }
            else:
                scores = (
                    candidate.get("scores", {}) if isinstance(candidate.get("scores"), dict) else {}
                )
                result["retrieval"] = {
                    "query": query.strip(),
                    "status": "observed",
                    "semantic_rank": scores.get("semantic_rank"),
                    "semantic_similarity": scores.get("semantic"),
                    "graph_activated": bool(scores.get("graph_energy", 0)),
                    "in_semantic_candidates": scores.get("semantic_rank") is not None,
                    "explanation": ["Scores recorded by the actual retrieval request."],
                    "scores": scores,
                }
        return result

    def run_consolidation(self) -> dict[str, Any]:
        try:
            result = self._service_request("maintenance", {"context": self._context_payload()})
            report = {
                "present": True,
                "status": (
                    "pass"
                    if result.get("status") in {"committed", "ready", None}
                    else result.get("status", "warn")
                ),
                "triggered_by": "manual",
                **result,
            }
            self._last_consolidation_report = report
            return report
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
        try:
            context = self._context_payload()
            probe = self._service_request("probe_cognition", {"context": context})
            health = self._service_request("health", {})
            steps = {
                "input_specific_extraction": (
                    "pass" if probe.get("input_specific_extraction") is True else "fail"
                ),
                "service": "pass" if health.get("status") == "running" else "fail",
            }
            status = (
                "fail"
                if any(value == "fail" for value in steps.values())
                else "warn" if any(value == "warn" for value in steps.values()) else "pass"
            )
            report = {
                "status": status,
                "steps": steps,
                "probe": probe,
                "scope": "temporary source extraction; no accepted memories written",
                "host_capture": "not_observed",
                "restart_persistence": "not_observed",
            }
            self._last_smoke_report = {"present": True, **report}
            return report
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

    def remember_memory(self, content: str, *, scope: str = "project") -> dict[str, Any]:
        return self._service_request(
            "remember", {"context": self._context_payload(), "content": content, "scope": scope}
        )

    def retrieve_memory(
        self, query: str, *, profile: str = "baseline", max_context_tokens: int = 1200
    ) -> dict[str, Any]:
        return self._service_request(
            "retrieve",
            {
                "query": query,
                "context": self._context_payload(),
                "allowed_scopes": ["project", "workspace", "global"],
                "top_k": 10,
                "max_context_tokens": max_context_tokens,
                "profile": profile,
            },
        )

    def correct_memory(self, fact_id: str, content: str, expected_version: int) -> dict[str, Any]:
        return self._service_request(
            "correct",
            {
                "context": self._context_payload(),
                "fact_id": fact_id,
                "content": content,
                "expected_version": expected_version,
            },
        )

    def forget_memory(self, fact_id: str, expected_version: int) -> dict[str, Any]:
        return self._service_request(
            "forget",
            {
                "context": self._context_payload(),
                "fact_id": fact_id,
                "expected_version": expected_version,
            },
        )

    def get_dashboard_view(self) -> dict[str, Any]:
        resolved = self._resolved_settings()
        health = self.health_report()
        memory = self.get_memory_snapshot(limit=8)
        scope = self.get_scope_status()
        current_host = resolved.settings.onboarding.preferred_host
        next_actions = list(health.get("next_actions", []))
        current_host_entry = _host_entry(current_host)
        if current_host_entry["tier"] != "Tier 1":
            next_actions.append(
                f"Validate {current_host_entry['label']} in a real session; adapter tests alone do not establish host continuity."
            )
        if self._last_smoke_report is None:
            next_actions.append(
                "Run the smoke check to verify input-specific extraction. Host capture and restart behavior need separate validation."
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
                "active_sqlite_path": health.get("service", {})
                .get("runtime", {})
                .get("database_path"),
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
            "scope": scope,
            "profile": scope.get("profile"),
            "jobs": memory.get("jobs", {}),
            "health": health,
            "activity": {
                "last_smoke_test": self._last_smoke_report or _default_activity(),
                "last_consolidation": self._last_consolidation_report or _default_activity(),
            },
            "next_actions": deduped_actions,
        }
