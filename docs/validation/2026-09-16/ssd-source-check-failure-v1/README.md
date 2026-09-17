# SSD source verification: missing external website fixture

Frozen source `82b7116` completed 864 passing tests, four skips, and one failure
in 188.29 seconds. Fresh source and temporary fixtures were on C:. The failed
documentation test looked for `web/index.html` above the relocated checkout;
that separate website checkout is on D:. All three continuity cases passed.
Formatting, typing, JavaScript, build, and Twine did not run after pytest failed.

The repair allows an explicit `MNEMOS_WEBSITE_INDEX` path without changing any
documentation assertion. All eleven documentation tests passed with that path.
The next full runner will record and compare the external file hash before and
after verification. This failed full run remains preserved; the focused pass
does not substitute for a full check or installed-package validation.
