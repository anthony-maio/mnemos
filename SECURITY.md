# Security Policy

## Supported scope

Mnemos is a local-first memory system. Security issues are most likely to involve:

- accidental secret capture
- unsafe memory persistence
- store corruption or unsafe deletion paths
- MCP exposure or host integration mistakes

## Reporting

Please report security issues privately to the maintainers before opening a public issue. Include:

- affected version or commit
- reproduction steps
- expected vs actual behavior
- whether secrets, personal data, or memory integrity are affected

## Handling guidance

- Do not publish credentials, tokens, or private user data in reports.
- If the issue involves captured memory, describe the pattern rather than pasting sensitive content.
- If the issue affects scope isolation, include the exact scopes and scope IDs involved.

## Current hard boundaries

- Mnemos should not be used to intentionally store secrets.
- `SimpleEmbeddingProvider` is for development, not a hardened production provider.
- Remote/shared multi-user memory is out of scope for v1.
