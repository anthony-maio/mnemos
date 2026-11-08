# Local 0.7.0 FastEmbed comparison

Compared the preserved original checkout at 5fcc925d with its existing edits
against delivery de45f13b92b3bf7a43bd65c9220ec9fe71b154d4. The receipt records
hashes of embeddings.py, settings.py, runtime.py and pyproject.toml in both trees.
Those hashes remained unchanged through the comparison.

FastembedProvider and _default_embedding_model are AST-identical. The embedding
portion of AppSettings.apply_inferred_defaults is unchanged; delivery adds
judgment model defaults after the embedding defaults. Delivery selects FastEmbed
for a fresh configuration and retains explicit provider/model choices.

All 13 tests in delivery tests/test_embedding_defaults.py passed with isolated
user-state fixtures and a fresh external pytest temporary directory. These use
a stub provider to test selection across CLI, MCP and service entry points.
They do not measure ONNX retrieval quality, first-download behavior or latency.
The separately retained installed-package smoke test covers actual FastEmbed
restart recall; it is not a general quality or performance benchmark.

No FastEmbed provider port is needed. This closes the missing-implementation
comparison for that feature. Automatic priming, first-run UI and durable-pipeline
migration decisions remain separate. No source, profile, hook configuration or
live database was changed by this comparison.

Source hashes and the test receipt hash are recorded in
`local-070-fastembed-comparison-v1.json` alongside this report. The original
private JUnit is `.venv/local-070-fastembed-v1.xml` in the improvement worktree.
