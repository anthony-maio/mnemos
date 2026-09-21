# Source evidence mutation guard

The production extraction path copies source events before constructing its
prompt and compares the caller-owned evidence after awaiting the provider.
Changed source content, scope, role or list membership rejects the response,
including otherwise valid no-facts output. Correction authority uses the copy.

Five new cases failed before the fix and pass afterward. The broader focused
suite passes 62 tests with zero failures, errors or skips; its JUnit report is
preserved here. Formatting and Python 3.10 mypy passed in the execution session.
Those console results are observations, not separately archived raw logs.

This checks concurrent caller mutation with deterministic providers. It does
not establish model extraction quality, full-module acceptance or a native-host
failure reproduction. Full source/package and installed checks remain pending.
