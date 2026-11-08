# Relation inventory v2 development execution

Execution follow-up: the fixed run stopped on its first uncertain completion and failed the development screen. It is closed without retry or regrading. See `docs/validation/2026-11-08/relation-inventory-execution-v2.md` for results, completed-output review and cleanup evidence. The prospective protocol below is retained unchanged.

Status: protocol fixed before semantic execution. V2 source tokenizer preflight passed all 127 prompts at 1,521-1,609 tokens with 2,048 output tokens reserved; no semantic calls were made during preflight. Its parent/native processes are absent, ports are closed and bindings are unchanged. Wrapper session 27068 ended with exit code zero. The independent audit verified the journal and all source token records; evidence hashes are in `docs/validation/2026-11-08/relation-inventory-tokenizer-v2.json`. Dynamic query token checks remain part of execution. The source/query harness passed 136 model-free checks. This run is development evidence and cannot satisfy fresh acceptance or actual pilot requirements.

## Frozen candidate and cohort

Use the prepared manifest and policies in `.venv/relation-inventory-development-v2`. The manifest contains the original 68 questions and 127 candidate observations without filtering competitors. Its SHA-256 is `ce763184577d0306ab1fa528387440b0e9d075c19d3629e9d6289489b98e9a83`. Policies SHA-256 is `02fd3b534c5a7684127e53459975b45af1f539de81de48cf2a63767c814f538d`.

Bind original answer labels at SHA-256 `f2af04ad22b277dc096d9d084b85b871816e514814739673fcf52ed9f790e916` and source-grading targets at SHA-256 `29195a64ed1b9291a8daf111c396ee4d531423f75812069a182f778bc314e9f9`. The runtime wrapper hashes these files but never passes their contents to either producer. Source grading uses the frozen whole-fact targets and manual semantic rubric. Record semantic review decisions with source-based rationales; schema validity and quotation matches earn no automatic semantic credit.

Use the retained Qwen3:14b model and Ollama runtime bound by the policy. Both stages use an 8,192-token context, 2,048-token output limit, temperature zero, seed 2026110806, raw rendering and disabled thinking. Each producer has a 15-second cooperative deadline including transport token checks. There are no generation warmups. The first source request includes first-generation effects after the empty model load; report that limitation rather than labeling it warm performance.

## Request schedule and stopping

Run case keys in frozen lexical order. In the first pass, annotate each distinct source once before its first query. Completed malformed annotations remain unavailable and are never retried. The reuse pass issues query interpretation only, using the same cache. Maximum generation requests are 127 source requests plus 136 query requests; missing inventories may reduce actual query requests but never reduce scoring denominators.

Any transport failure or unconfirmed completion stops both passes without retry. The wrapper tokenizes every rendered query immediately before generation because inventories determine its final size. A request exceeding context plus reserved output stops without generation. The generic harness conservatively records transport preflight rejection under its unconfirmed-completion stop state; the wire journal and completion counter distinguish known-unsent requests from potentially completed generation. Preserve that distinction in the report.

No deadline extension, prompt change, answer-label adjustment, retry or new threshold is allowed within this run. Any later candidate requires a separate version and evidence directory.

## Runtime and evidence

The prepared wrapper is `.venv/run-relation-inventory-v2.py`. It reconstructs and verifies the complete manifest before starting an isolated Ollama server on loopback port 11437. It checks that the port is free, disables cloud access and uses the retained local model directory. It verifies parent and native child ownership, model path and runtime hashes. It rechecks all prepared source token IDs against the completed v2 tokenizer preflight before semantic execution.

The prepared execution-wrapper SHA-256 is `68d34e520fb194294e60711ded5fe1e09c18949290a2669133df9bec35f9434a`.

Before runtime startup, `freeze.json` binds the wrapper, ownership helper, model blob, executable/native libraries, producer code, policies, input files, grading files, dependency versions and tokenizer evidence. Source tokens and semantic events are written to separate exclusive journals. Semantic events retain unrendered and rendered prompts, token IDs, full provider replies, completion errors, decisions, per-claim temporal support bases and unattempted query identifiers. A journal failure aborts execution.

On exit, unload the model and stop only the verified owned process tree. Verify parent/native process absence, closed ports and unchanged runtime/input bindings. A failed or interrupted run remains preserved; do not delete or reuse its directory.

## Measures and limits

Report source coverage, whole-fact support accuracy and manual semantic dimensions separately from final precision, critical retention, recall, abstention and availability. Unavailable or unattempted queries get no successful-abstention credit. Current/date questions cannot fall back to source-state intent. Preserve the basis for every selected claim.

First-pass case timing includes annotation work awaited by that query. Query timing includes inventory lookup, query tokenization and interpretation. The reuse pass measures cached annotation behavior. Neither timer includes initial retrieval, and source timing includes evidence-journal costs. These measures do not prove the original full 500 ms retrieval or 100 ms durable capture gates. The first pass has 127 distinct source identities and therefore no cross-question annotation reuse.

Original quality gates, fresh evaluation, actual daily use and release/publication requirements remain unchanged.
