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

    def get_settings_view(self) -> dict[str, Any]:
        resolved = self._resolved_settings()
        settings_payload = resolved.settings.model_dump(mode="python", exclude_none=True)
        return {
            "settings": _mask_secret_fields(settings_payload),
            "paths": {
                "global_config": str(resolved.global_config_path),
                "project_config": (
                    None
                    if resolved.project_config_path is None
                    else str(resolved.project_config_path)
                ),
            },
            "warnings": resolved.warnings,
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
            "settings": _mask_secret_fields(
                imported.settings.model_dump(mode="python", exclude_none=True)
            ),
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
            return {
                "count": len(chunks),
                "recent": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "scope": chunk.metadata.get("scope", "global"),
                        "scope_id": chunk.metadata.get("scope_id"),
                        "updated_at": chunk.updated_at.isoformat(),
                        "access_count": chunk.access_count,
                    }
                    for chunk in recent
                ],
            }
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def get_memory_detail(self, chunk_id: str) -> dict[str, Any]:
        resolved = self._resolved_settings()
        engine = self._build_engine(resolved.settings)
        try:
            payload = build_chunk_inspection(engine, chunk_id)
            if payload is None:
                raise KeyError(chunk_id)
            return payload
        finally:
            close = getattr(engine.store, "close", None)
            if callable(close):
                close()

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

                await engine.consolidate()
                steps["consolidate"] = "pass"

                health = self.health_report()
                steps["health"] = "pass" if health["status"] == "ready" else "warn"
            finally:
                close = getattr(engine.store, "close", None)
                if callable(close):
                    close()

            status = "fail" if any(step == "fail" for step in steps.values()) else "pass"
            return {"status": status, "steps": steps}

        try:
            return asyncio.run(_run())
        except Exception as exc:
            return {
                "status": "fail",
                "steps": {
                    "store": "fail",
                    "retrieve": "fail",
                    "consolidate": "fail",
                    "health": "fail",
                },
                "error": str(exc),
            }
