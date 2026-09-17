# Retrieval during a concurrent correction or forget

Source `62fe16419d10a8c0d89718ffbf9767f0ada41933` repairs a managed-service
interleaving in the biomimetic retrieval path. Candidate records were selected
before awaiting state classification. A correction could commit during that
wait. Retrieval then scored and budgeted the old content but returned the new
record from its final storage lookup. A forget could remove the selected fact
and incorrectly hide an otherwise eligible fallback.

The regression uses SQLite and real service dispatch. It pauses only the
classifier with an asyncio event, commits a correction through `correct` and
maintenance (or dispatches `forget`), then resumes retrieval. It covers an
irrelevant replacement, an over-budget replacement, and forgetting, each with
graph expansion enabled and disabled. The prior frozen `8d865a5` source fails
all six cases; the replacement case returns 2,150 bytes against a 300-byte
context bound. The repaired source passes all six, returns the eligible
fallback with its actual semantic score, and does not record an extra retrieval
for the excluded fact.

The repair reloads semantic and lexical candidates after classification, before
graph expansion, scoring, and budgeting. The managed owner-loop path has no
further await between that refresh and the response. Baseline retrieval is
unchanged, and an empty candidate set still avoids a classifier call. This does
not claim snapshot isolation against an unsupported external database writer.

The first four cases failed before implementation. A subsequent test correction
replaced an incorrect zero-access-count assumption with comparison to the
post-correction count; correction itself legitimately increments that counter.
The six-case comparison against the old frozen source ran after implementation.
The related retrieval, service, and graph selection passed 93 tests before the
two forget cases were added. Strict typing and formatting checks passed.

Full source and installed-package verification for this repair is tracked
separately. The active real-model queue retains its original frozen source and
does not measure this later repair. No live database or host settings changed.
