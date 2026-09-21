# Evaluation provenance traversal

The provenance scanner previously traversed every directory before filtering
out files below .git, .venv and __pycache__. The repaired scanner prunes those
same excluded names before descent. It retains the existing eligible Python
and TOML paths, sorting and path/content hash encoding. Validation archives
remain included when they contain eligible source files; this is not a silent
change in what the code hash covers.

A guarded-directory regression reproduced traversal into an excluded .git
subdirectory before the fix. On Python 3.13 the test must guard both os.scandir
and pathlib's cached globber entry; guarding os.scandir alone initially missed
the traversal. Exact fixture hash parity and an unreadable included directory
are also tested. Included-directory scan errors now fail closed instead of
allowing an incomplete provenance result.

After the repair, 57 focused and adjacent tests passed in 47.03 seconds across
provenance traversal, thinking options and the local-memory benchmark tests.
Black and Python 3.10 mypy passed. This is not a controlled speed benchmark or
a new full-suite result. The independently frozen assertion-inventory model
run still uses its unchanged source snapshot and indexed files.
