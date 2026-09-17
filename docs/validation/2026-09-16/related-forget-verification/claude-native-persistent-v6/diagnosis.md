# Claude persistent v6 duplicate diagnosis

Status: the v6 native run completed the model turn and durable restart checks, but was correctly marked not_observed. The preserved report is `D:\WindowsTemp\mnemos-native-claude-persistent-v6-ccg6jp_b` and the sanitized report is `.venv/native-claude-persistent-v6/report.json`.

## Observed sequence

The private hook traces contain one SessionStart, one UserPromptSubmit, and two Stop records. Every adapter subprocess returned zero and completed. UserPromptSubmit and both Stop records reported `degraded`; this is expected when the native hook payload has no `prompt_id`.

The transcript contains one user record and two assistant records. The first assistant record has no usable content and the second has the answer. The user record has a native UUID digest `2de601dfe0f62c4c5918d85246d6796564a83207e7d6d40791728e818d0dff46` and a transcript `promptId` digest `56238c7f710db8b827c58e45934e1f67b1a3664df6603bde785752786edc151b`. The hook payload contains neither `prompt_id` nor `promptId`.

SQLite contains three post-seed captured rows:

- A user row for the fallback prompt, with source reference `prompt:<prompt-content-digest>` and a `claude:v2` event ID.
- A second user row for the same normalized prompt, with a native transcript UUID source reference and a different `claude:v2` event ID.
- One assistant row with a native transcript UUID source reference and a `claude:v2` event ID.

The fallback user row was created before the transcript-backed user row. The checkpoint has one pending user marker with the prompt fallback source reference and a null turn reference. Its offset is 9503, and its last-assistant marker is present. The second Stop did not add another row: replay left the SQLite event count unchanged at four and advanced no new event identity.

## Cause

This is the existing conservative ambiguity path, not a new data-loss bug. The transcript reader accepts both `prompt_id` and `promptId` at `mnemos/adapters/claude_code.py:189-200`, so it obtains the native transcript turn reference. The UserPromptSubmit path only reads `payload["prompt_id"]` at `mnemos/adapters/claude_code.py:675-684`; because the host omitted that field, it creates a content-hash fallback source reference and records a pending marker with `turn_ref=None` at `:692-702`.

The pending matcher intentionally requires a nonempty transcript turn reference equal to the pending turn reference at `mnemos/adapters/claude_code.py:295-303`. The native transcript has a turn reference, while the fallback marker does not, so the existing fallback cannot be safely associated with the transcript UUID. `_capture_transcript` therefore keeps the pending marker, captures the transcript-backed user evidence separately, and returns `AMBIGUOUS_TRANSCRIPT_ASSOCIATION`. The Stop path preserves that degraded result at `:737-752` and does not deduplicate on text alone. The result is two user rows with distinct source identities, while the assistant transcript row is retained.

The two Stop traces are not two assistant captures. The first processes the transcript delta and records the assistant marker. The later Stop sees no new deltas and returns the existing ambiguous replay state. The unchanged replay event count and nondecreasing checkpoint offset support this interpretation.

## Safety assessment

The duplicate is observable and honestly degraded. It does not discard the assistant answer, overwrite the native UUID, or claim a clean capture. Matching only the prompt text here would be unsafe because repeated genuine turns may have identical text. The current behavior preserves both source identities and leaves an explicit pending marker for review.

## Minimal no-model regression

Use a disposable real store and a fake Claude client, with no model calls:

1. Send `UserPromptSubmit` with a prompt and transcript path but no `prompt_id`; assert one fallback user event, degraded status, and a pending marker with null `turn_ref`.
2. Append one transcript user record with the same content, a UUID, and a distinct `promptId`, followed by one assistant record; send `Stop`; assert both user events remain, source references differ, the transcript-backed user source is the UUID, the assistant is captured, and Stop remains degraded.
3. Repeat the same content in two distinct transcript user records with different UUID and `promptId` values; assert neither transcript record is collapsed into the other and all native UUID-backed records remain distinct.
4. Replay the unchanged Stop payload after reopening the store; assert no additional event IDs and a nondecreasing checkpoint offset.
5. Add a positive case where UserPromptSubmit and the transcript use the same explicit prompt ID; assert the fallback and transcript delta associate once without a duplicate.

A future repair should require a stable host turn association, or a separately proven transcript parent/turn mapping, before suppressing the fallback. It must not treat a unique content match as sufficient when multiple identical turns are possible.

