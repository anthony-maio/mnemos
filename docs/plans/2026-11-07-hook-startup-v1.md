# Isolated command startup diagnostic

Measure the existing installed scoped ranking candidate at `5735c502` with
normal isolated Python startup and with isolated startup that skips `site`.
Use the same native validation interpreter and installed package target as the
complete installed comparison. Do not change product files or host launchers.

There are four arms: empty command with site, empty command without site,
resident-client imports with site, and resident-client imports without site.
The client arms import the installed hook client, bounded input reader and
diagnostics module. The empty arms omit those imports. All arms use the same
entry file and record elapsed import time separately from complete subprocess
time. The output receipt and its required stdlib imports remain inside the
complete timer; neither timer is a complete-hook acceptance measurement.

Run two warmup rounds and 20 observed rounds. Rotate the first arm each round
in a fixed four-arm cycle. Retain all 88 observations and 80 timed samples.
Record interpreter/prefix identity, site initialization state, imported module
origins and an environment hash, never environment values. Verify installed
package files against the wheel before and after measurement. Freeze protocol,
driver, entry file and package bindings before the first subprocess.

After timing, run one separate import-time trace for each client arm. These
instrumented traces are diagnostics only and never enter the latency summary.
No service, model, hook dispatch, database or live configuration is involved.

Recompute the counts, medians, interpolated p95, maxima and matched round
differences independently. Compare the environment fingerprints across modes.
Skipping site may omit dependency setup and venv initialization, so a fast
import does not prove fallback behavior. Any later product change must restore
the original runtime and private-launcher checks before an ordinary fallback,
preserve safe refusal versus unknown outcome, and pass complete installed
workflow validation. Keep the 100 ms hook and 500 ms recall requirements intact.
