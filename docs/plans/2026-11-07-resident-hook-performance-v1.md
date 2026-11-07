# Installed resident hook comparison

Compare the ordinary command and the opt-in resident adapter from the same
installed wheel, built at `feb6ac3b3eebd2e178ec584b3be1e72cca1228fc`. Keep the
100 ms complete-hook and 500 ms recall p95 limits. This is a performance and
durability experiment on synthetic data, not native-host or pilot acceptance.

Use a new private directory on C:, the prior installed-performance Python and
dependency versions, and the verified resident-worker wheel installed in its
separate target. Bootstrap that target first under isolated Python. Verify all
loaded Mnemos modules come from it. Preserve the original V3 run and its files.

Rebuild the original 10,000 accepted facts with their original source evidence,
project identity and real BGE-small-en-v1.5 document embeddings. Use the same
cached model assets and compare every accepted vector against V3. Start with
zero capture jobs; retain a pristine database snapshot. Do not copy V3's later
unknown captures into the new fixture. Keep normal maintenance enabled.

The ordinary command runs the unchanged private launcher. The resident command
uses the exact environment setup statements extracted from that launcher's
`prepare` function. Validate their AST equivalence before use. Apply the full
original `prepare` at worker startup. The worker reloads and compares effective
settings, including explicit LLM selection, on every event. This comparison is
inside the timer. Its environment and working-directory checks also remain.

For a safe fallback, the resident command must call the full original `prepare`
before the ordinary adapter. Its private integration wrapper installs that
checked fallback into the installed command module. Test refusal of an explicit
LLM configuration, safe fallback with an absent worker, and the lightweight
successful command path before measurement. Do not remove launcher checks or
modify installed product files. No live host configuration uses this wrapper.

Freeze the protocol, driver, wrappers, original helpers, source, wheel, installed
package files, dependencies, model assets, input manifest, settings, registry,
initial database and accepted cohort before timing. Record only the hash of
the effective environment, never its values. Preserve preparation failures.

Run three recall warmups and 50 timed recall requests using one normal resident
client. Run one warmup per event and command arm, then 50 paired measurements
for each of Stop and UserPromptSubmit. Alternate which arm runs first in each
pair. Pair inputs have identical text and distinct source identities. Use the
same memory service and accepted cohort for both arms. Unknown new captures
must not become accepted facts. Record all attempts, failures and warmups.

Time each complete subprocess from before launch through exit, including Python
startup, environment setup, configuration checks, authenticated worker dispatch,
durable capture, real retrieval and output. Require committed source/job records
and nonempty qualified prompt context; compare paired host output exactly. Check
that successful resident commands did not import the ordinary runtime or take
fallback. Preserve stderr and raw outputs privately. No startup subtraction,
component timer, disabled retrieval or cached answer substitutes for this gate.

Supervise the memory service and worker separately with the verified Windows
launcher/interpreter ownership helper. Record startup and ownership-check time
separately. Register processes immediately, stop only verified owners, and prove
all sources and jobs survive service restart. A forced stop fails cleanup.
Afterward verify frozen inputs, accepted facts/evidence/vectors, package/model
bytes, process absence and descriptor removal. Report medians, interpolated p95,
maxima, operational failures and paired differences. Keep the pilot inactive.
