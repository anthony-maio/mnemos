# Source-bound recovery: bounded lifecycle passed

Frozen source 11a97e34214a038e208c116d0778d0dc5681fcff passed the original
outage fixture's unchanged lifecycle assertions using local Qwen3.5 9B and
Nomic embeddings. Both pending jobs completed on attempt four. Extraction
retained one exact user-source marker, excluding the adjacent one-turn tool and
reply instructions. The assistant acknowledgment was filtered by surprisal.
Manual comparison is recorded in semantic-review.json; the runner's original
semantic-review-pending label remains unchanged as historical raw evidence.

Scoped recall found the marker in its project and returned no foreign-project
facts. Explicit correction retained the fact identity, incremented its version,
and survived service restart. Related forgetting followed by original-outbox
replay, maintenance and another restart left no facts. The service removed its
runtime descriptor. The archived SQLite backup passes integrity and foreign-key
checks. Runtime credentials and descriptors are excluded.

All five module flags were enabled. This one-fact fixture does not establish
useful graph traversal, affective calibration, automatic correction quality or
module superiority. It is a frozen-source service run with real local providers,
not an installed-wheel check, new native-host session, held-out evaluation,
full ablation or pilot. The earlier zero-fact and harness failures remain
preserved. No evaluation threshold or lifecycle assertion was weakened.

Exact model digests, provider options, source hashes and configuration are in
preparation.json. The frozen source ZIP is in
../source-bound-recovery-preparation-v1/source.zip. No live configuration or
memory data changed.
