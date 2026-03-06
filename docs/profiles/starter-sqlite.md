# Profile: Starter (SQLite)

Use this for zero-friction install and immediate persistence.

## When to Use

- Single-user local memory
- Plugin-first onboarding
- No external services required

## Environment

```bash
MNEMOS_STORE_TYPE=sqlite
MNEMOS_SQLITE_PATH=.mnemos/memory.db
MNEMOS_LLM_PROVIDER=openclaw   # or openai/ollama/mock
```

## Notes

- This is the default for plugin installs.
- Run `mnemos-cli doctor` to verify readiness.
- Recommended next step for local production: embedded Qdrant profile.
