# Frozen service/runtime checkpoint failure

Source `e1fd56e36c17f1b1b05ead6c579619cb5ad372dd` was checked from a clean
checkout on C: with fresh C: test fixtures outside Git. The full suite finished
with 897 passed, one failed, and four skipped in 460.15 seconds.

The Codex parameter of the cross-host continuity test failed when a list
response lacked the `facts` field after explicit remember and maintenance.
The original output does not print that response, so its precise cause is
not established by the KeyError. The test uses synthetic providers; this is
not a native Codex session or a real-model quality failure.

The checkpoint runner stopped at pytest. Its later formatting, typing,
JavaScript, build, Twine, and installed-wheel steps did not run. Earlier focused
passes remain separate evidence and do not supersede this failure. The
original report, log, JUnit XML, and executed runner are preserved unchanged.
No timeout, durability, or acceptance threshold was changed for this attempt.
