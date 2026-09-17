# Capture and forget overlap validation

The [corrected v2 probe](v2/README.md) uses the same assertion in both records
and demonstrates the retained-copy limitation. The files at this directory's
top level preserve the earlier v1 fixture described below.

This v1 artifact is retained as an invalid-fixture control. It is an offline development diagnostic against frozen source commit `9e253390c58302817a18b6504f55899b36a2c27b`. The helper used a disposable SQLite database, `UnavailableLLMProvider`, `SimpleEmbeddingProvider`, and a registered enabled project. No live database, model endpoint, host, or configuration was used.

The archived executable is stored as `capture_forget_overlap_probe.py.txt` so repository Python checks do not execute validation artifacts.

The fixture disabled all five module switches so that the capture job promoted the captured prompt verbatim. That is an intentional raw capture ablation. `ModuleConfig` defaults all five switches to true in the product configuration; retrieval's `baseline` profile is a separate ranking choice. This artifact does not claim that the default cognition path promotes every captured prompt.

The prompt became one fact with `source=sleep_consolidation`. An explicit remember became a separate fact, which was corrected to version 2 and forgotten. Retrieval after forget, including after SQLite restart, still returned the prompt fact because it had separate evidence and was outside the selected fact's source and canonical suppression records. The prompt and forgotten fact shared no token or content, so this is an unrelated-fact control and cannot support an overlap or information-leak claim.

The v1 result is an unrelated-record preservation control. It does not
demonstrate failure to remove a matching assertion.
