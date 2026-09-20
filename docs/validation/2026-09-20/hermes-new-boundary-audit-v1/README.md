# Actual Hermes new-session boundary audit

Archive verification failed: the index includes two transient SQLite sidecars
that disappeared after the collector exited. Main database hashes match.
See verification-failure.json and the [corrected archive](../hermes-new-boundary-audit-v2/README.md).
The original index remains unchanged.

This archive preserves two independent probes of the actual
CLISessionMixin.new_session method. Both use private SessionDB files, actual
AIAgent persistence/evidence methods, MemoryManager, and the installed Mnemos
provider. The interactive constructor is bypassed to avoid credential and
tool discovery. UI notifications and model-selection configuration are test
scaffolding; persistence, evidence loading, provider delivery, and session
switching are not replaced.

## Missing uncommitted message

V9 fails: the old session starts with no durable messages and the in-memory
history contains one uncommitted user message. The actual new-session handler
passes that entire history as the already-persisted conversation prefix. The
flush collector skips it, leaving zero durable rows. The assertion failure,
original helper, and independently inspected SQLite backup remain preserved.
This is an open capture gap, not a passing boundary result.

## Committed evidence and real projection failure

V10 first persists the message through the actual agent flush method, then
calls the actual new-session handler. The valid case queues one capture with
the original session and native row ID, and the provider switches to the new
session. In the second case, the probe deliberately changes the persisted
evidence component's content. The real projection loader rejects its mismatch
with the original message. The repaired handler supplies authoritative empty
evidence, queues zero captures, and still switches the provider to the new
session. No evidence loader or commit method is mocked in either case.

All 55 installed Python modules match the verified Mnemos wheel byte for byte.
The wheel SHA is c5452e7b6810da567fc9a8ae9e46af9c0d96280bc0b78653218b17d6e6a4235e.
The candidate is based on Hermes 14a346345486418aa85ed6526549412ee2665ade;
its full tracked patch matches the digest recorded in V10. This remains an
uninstalled compatibility candidate. These probes verify outbox receipt,
not service delivery, service-side suppression, or the full interactive UI.

## Broader host tests

The initial full SessionDB run records 1,398 passes, 11 skips, and five
failures. Four subprocess tests could not import their YAML dependency after
resetting PYTHONPATH. With the private test environment repaired, the fresh
run records 1,402 passes, 11 skips, and one failure. The 42 affected boundary
tests pass separately.

A clean checkout of the unmodified Hermes base also records 1,402 passes,
11 skips, and the same failure: the concurrent WAL test rejects deleted
WAL/SHM descriptors left by preceding quarantine tests. The isolated test
passes. This establishes that the observed full-suite failure is also present
without the provenance patch; it does not make either full-suite run green.
The assertion and quarantine behavior have not been weakened.

The index binds 25 artifacts, including raw failures, successes, the candidate
patch, helper code, and three integrity-checked synthetic database backups.
This README is outside the index. The earlier helper-only and injected-error
probes are not substituted for these actual-handler observations.
