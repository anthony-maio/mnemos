# Support

## Where to start

Use the GitHub issue templates for public support traffic:

- bug report
- feature request
- host compatibility report

Read [docs/public-release-package.md](docs/public-release-package.md) and [docs/client-compatibility-matrix.md](docs/client-compatibility-matrix.md) first if you are evaluating Mnemos for adoption.

## Before opening an issue

Collect the minimum diagnostic set:

- `mnemos-cli doctor`
- `mnemos-cli stats`
- storage backend in use
- LLM and embedding provider configuration
- whether the issue reproduces on SQLite starter profile

## Good issue reports

Include:

- exact command or host workflow
- issue template used
- relevant scope and `scope_id`
- expected result
- actual result
- minimal reproduction steps

## Common triage buckets

- memory correctness or scope leakage
- retrieval quality or latency
- MCP host integration
- startup or persistence failures
- safety / redaction behavior

## Release expectations

If you are evaluating Mnemos for production, treat [docs/release-checklist.md](docs/release-checklist.md), [docs/public-release-package.md](docs/public-release-package.md), and [docs/client-compatibility-matrix.md](docs/client-compatibility-matrix.md) as the source of truth for what is verified today.
