# Opt-in source-bound service integration

The candidate on 01d3bc6 passed 211 focused tests, including real SQLite service
commits for additions, support without duplication, corrections, mixed and
unresolved-only results, disabled-module behavior and source mutation during
embedding. The candidate was integrated as de77de0 with identical Git blobs for
product, benchmark and test files. Nine promoted implementations match their
previous benchmark bodies after line-ending normalization.

Python 3.10 mypy and Black passed for 18 files. The wheel built successfully.
Its product extractor imported directly from the wheel with benchmark imports
blocked, and all eleven extraction-package files matched source bytes. The
archived wheel belongs to the isolated candidate; this zip-import smoke is not
a complete installed-wheel lifecycle check. Console type/format/build outputs
are observations rather than separately archived logs.

Settings retain generated extraction as the default. Explicit source_bound mode
uses the promoted two-stage extractor and durable review transaction. Disabling
mutable RAG defers automatic corrections as reviews; disabling sleep retains
capture passthrough. Cognition probing follows the selected mode. The final
transaction compares the original source snapshot, including accepted facts
that awaited document embedding.

No real-model evaluation, live configuration change, live migration or release
publication occurred. Review inspection/resolution/discard controls remain open,
as do current full-suite/package acceptance, real-model lifecycle recovery,
held-out ablations, native-host acceptance and the daily-use pilot. Earlier
small development results do not establish quality for this integrated path.
