# Claude degraded diagnostic reasons

Claude hooks previously reported only `status=degraded`, even when the adapter knew that a native occurrence ID was missing or a transcript association was ambiguous. The hook now passes its error through an exact allowlist. Recognized identity, transcript, capture, and routing codes appear as `reason`; other strings become `unknown`. Exception messages and captured text remain excluded from stderr.

Two regression cases run the real adapter through its command entry point with a fixture client. Missing-ID prompt capture remains provisional; the later native user and assistant entries remain separate, and the unresolved pending association stays in the checkpoint. Degraded status and the host stdout protocol are unchanged. Privacy cases reject arbitrary errors, recognized-code prefixes with secret suffixes, and multiline errors.

All 68 focused Claude, Codex, and import-boundary tests passed. Strict mypy passed for 50 product files, and the final formatting check passed. `pytest.xml` is the original machine report; `verification.json` records the commands and observed terminal results. `source.patch` binds this repair to the recorded base commit. Final test formatting changed only line wrapping after pytest.

This repairs observability. It does not resolve missing native IDs or establish complete capture, native hook deadlines, or clean shutdown acceptance. The installed ebe1b0d wheel and frozen real-model run do not include this change.
