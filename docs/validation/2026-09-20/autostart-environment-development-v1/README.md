# Autostart environment and disposable storage repair

The [failed native outage attempt](../native-claude-outage-failure-v1/README.md)
exposed an ignored environment setting and missing host-side storage isolation.
Default clients now parse `MNEMOS_AUTO_START`; explicit library arguments keep
precedence. Bridge and managed CLI requests inherit that setting, while
`service-stop` always disables startup. Native phase environments separately
pin their SQLite and runtime paths inside the disposable root.

The first regression run fails 12 cases and passes 42. Eleven failures expose
the ignored setting or invalid-value handling; the remaining failure exposes
unbound storage. The first broader run passes 138, skips two, and fails one:
the test's subprocess guard also intercepted scope resolution's Git command.
That fixture was narrowed to the client module. The final focused run passes
139 tests with two platform skips, including 22 new client cases and one new
storage precedence case. Raw JUnit files preserve both failures and the pass.

Coverage includes environment spelling, invalid values, explicit overrides,
Claude prompt buffering, bridge capture, CLI status/stop, stale descriptors,
and queued replay to an existing service. The storage test resolves conflicting
global and project settings and verifies that disposable environment paths
take precedence. No native host or model is launched by these new cases.

These are source-level checks. Full source and installed-package validation,
followed by a separately versioned native outage attempt, remain required.
The failed attempt's runner and protocol stay frozen. No live configuration
or default-database cleanup was performed as part of this repair.

The [index](evidence-index.json) binds nine copied source/test artifacts.
