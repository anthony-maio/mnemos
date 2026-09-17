# Frozen service/runtime checkpoint failure

Source `33c6dd90128d8762053e0bf58862e4bc52484bab` was checked from a clean
checkout on C: with fresh C: test fixtures outside Git. The full suite finished
with 897 passed, one failed, and four skipped in 165.03 seconds.

The generated Codex PowerShell bootstrap test exceeded its existing ten-second
subprocess deadline. The synthetic hook output file was absent when checked
afterward. The earlier cross-host continuity failure did not recur, but its
original cause remains unproved. This is a generated-command subprocess test,
not a native Codex conversation or a real-model quality failure.
The checkpoint runner stopped at pytest. Its later formatting, typing,
JavaScript, build, Twine, and installed-wheel steps did not run. Earlier focused
passes remain separate evidence and do not supersede this failure. The
original report, log, JUnit XML, and executed runner are preserved unchanged.
No timeout, durability, or acceptance threshold was changed for this attempt.
