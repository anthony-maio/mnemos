# Affective parsing and Hermes write-guidance package checkpoint

Frozen source `8f8f63f3250858667dfe460bdd22295c279f2625` passes 1,324 source
tests with four skips, all six source/package checks, 543 isolated installed
cases with two matching skips, and the installed replay probe. The installed
run verifies each selected case and outcome against the source run, checks
module origins, and compares installed package bytes with the built wheel.

Wheel: 298,790 bytes, SHA-256
`53444f7f87141e7107fe9f4daf4a6095e04c9d69d007cd218634fd2fff50f99f`.

The affective parser now reads signed decimals and scientific notation without
turning digit fragments into measured states. Malformed or nonfinite triplets
remain unavailable and do not contaminate prior state history. Existing
finite-value clamping is preserved. The [parser diagnostic](../affective-response-contract-v1/README.md)
records zero changes when replaying 300 archived real-model responses; their
33/300 fidelity result remains unchanged.

The new parser regression was also executed after the repair against the
preserved old `68693f8` source. Six assertions fail and four pass there; all ten
pass in this checkpoint. `prior-parser-regressions` retains its exact report,
JUnit results, and output. This is a retrospective regression check, not a
claim that these parser assertion failures were recorded before implementation.

Hermes remember/correct descriptions now put the follow-up status call beside
the write operation and distinguish durable acknowledgement from processing.
The provider system block makes the same distinction. The three failed v3
native lifecycles remain failed; this package check does not demonstrate that
the revised guidance changes native model behavior. The prospective v4 cohort
uses the unchanged native acceptance criteria and records the supplied tool
descriptions. Live host configuration and memory stores remain unchanged.

The evidence index binds 37 files: source and installed reports/logs, the prior
parser regression, and exact executed verification helpers. It does not claim
complete host capture, module-quality acceptance, or operational pilot results.
