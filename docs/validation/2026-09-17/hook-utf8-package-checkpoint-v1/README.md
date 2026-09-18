# Hook UTF-8 package checkpoint

Frozen source `1068a629887ef9a5d267a19ead6eb64b2f125757` passed 944 tests with four platform skips, Black, strict mypy, JavaScript syntax, build, and Twine.

A fresh installed wheel passed 214 selected cases with two matching platform skips and seven replay assertions. Exact case identities, outcomes, and skip reasons match the frozen source run. The installed runner sets PYTHONSAFEPATH=1 so Python subprocesses do not import the source checkout merely because it is their working directory.

The new regression verifies exact Unicode capture from UTF-8 JSON through Windows-style stdin for both adapters, and nonblocking Codex Stop output after an exception. Native post-repair capture remains a separate check; real-model quality, latency, and pilot gates remain open.
