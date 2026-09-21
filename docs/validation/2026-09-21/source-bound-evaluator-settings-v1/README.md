# Source-bound evaluator configuration

The held-out launcher now accepts --extraction-mode generated|source_bound,
with generated retained as the default. The selected mode reaches each real
service factory and its provider trace and is included in the options hash.
Checkpoint resumption therefore cannot combine generated and source-bound runs.
Invalid extraction modes fail before discovery or output creation.

All seven profiles retain their original module switches in both modes. Tests
cover those 14 combinations, CLI propagation, configuration fingerprints and
existing evaluator behavior: 80 tests passed. Black passed. Python 3.10 mypy
passed for the two changed benchmark modules with artifacts.py explicitly
included to disambiguate the neighboring artifacts directory. The first mypy
invocation without that explicit module reported missing imported symbols.

This is harness verification, not an executed held-out model result. The fresh
v7 corpus and its frozen scoring criteria remain unchanged and unexecuted.
