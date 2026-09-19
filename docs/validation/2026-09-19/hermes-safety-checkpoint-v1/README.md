# Hermes safety package checkpoint

Source `51904d3abdf523e0afb086e956046ca1d5839e92` passes 1,170 tests with four
platform skips, Black, configured strict mypy, JavaScript syntax, package build,
and Twine. The isolated installed wheel passes 482 selected contract cases with
two matching skips and all seven replay assertions. The wheel is 293,018 bytes;
SHA-256: `c209536b5fab1fc75c8ef754ada317030baffffced00ebc21643271ee6e3f794`.

This checkpoint rejects missing Hermes session scope after failed initialization,
clears cached recall when scope is reinitialized, stops remove/replace callbacks
from being captured as new facts, and makes failed asynchronous captures visible
through flush/diagnostics. Two development tests preserve no-fact extraction and
accepted-fact correction accounting without changing evaluation criteria.

The first installed-run invocation supplied a directory where the verifier
requires a JSON report path. It stopped before installation. Its failure report
is retained separately; the corrected invocation used fresh output paths.

The 34 artifacts listed in `evidence-index.json` include source/package logs,
installed contract and replay results, bindings, and verification helpers.
This explanatory file is outside that index. The website input remained unchanged.

The [patched native run](../native-hermes-local-v3/README.md) subsequently found
a duplicate callback timestamp conflict. Therefore these passing package checks
do not establish complete native Hermes acceptance or final project completion.
