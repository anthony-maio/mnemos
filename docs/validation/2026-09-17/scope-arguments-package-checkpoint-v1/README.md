# MCP scope argument package checkpoint

Frozen source `30eaf4209bc6921b8a4a30f35ecd86fb68e1aae3` passed 929 tests with four platform skips, Black, strict mypy, JavaScript syntax, build, and Twine.

A fresh installed wheel passed 139 selected cases with two matching platform skips and seven replay assertions. Exact case identities, outcomes, and skip reasons match the frozen source run. The installed runner sets PYTHONSAFEPATH=1 so Python subprocesses do not import the source checkout merely because it is their working directory.

The MCP regression exercises both scope argument formats through stdio, rejects empty or invalid lists, and checks foreign-project isolation. These are package and protocol checks. Native Codex retrieval after the repair has not passed; real-model quality and pilot gates remain open.

The earlier complete test run failed typing because an archived test copy was treated as active Python. Its raw evidence is preserved in ../scope-package-archive-failure-v1. Archived test copies were renamed to .py.txt with identical bytes; typing rules were unchanged.
