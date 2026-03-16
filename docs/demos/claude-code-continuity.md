# Claude Code Continuity Demo

## Setup

- install `mnemos-memory[mcp]`
- run `mnemos ui`
- apply Claude Code host setup
- start a fresh Claude Code session with `mnemos-recall` available

## Side-by-side workflow

### Claude Code without Mnemos

Day two on the same repo usually starts with re-teaching context:

- which package manager this repo uses
- which deployment target is active
- what broke last time
- which preferences are repo-specific versus global

That costs prompt budget and makes continuity depend on what the model happened to keep.

### Claude Code with Mnemos

Day two starts with:

1. run `mnemos-recall` for the current repo scope
2. continue the task with scoped memory already surfaced
3. run `mnemos-curator` at the end to keep only durable facts
4. let consolidation compress session noise into longer-lived memory

What comes back is not "all prior chat history." It is the smaller set of durable repo facts that survived curation and consolidation.

## Example scenario

Repository facts already stored:

- `Use uv for Python tooling in this repo`
- `Production deploys run through fly.toml`
- `Last auth bug came from missing session cookie forwarding`

Next session prompt:

> fix the auth regression and prep a deploy

Expected Mnemos-assisted flow:

- `mnemos-recall` surfaces the `uv`, `fly.toml`, and prior auth-failure context
- Claude Code works from scoped repo memory instead of guessing or asking for re-explanation
- `mnemos-curator` stores only the new durable fact if something changed

## Why this is different from built-in memory

- stronger repo/workspace/global scope boundaries
- inspectable retrieval and chunk provenance
- explicit curator workflow instead of silent accumulation
- local SQLite storage with one-file persistence
