# Native Hermes fixed cohort v4: three failed lifecycles

All three scheduled attempts ran and failed. None was replaced or retried.
Runner `53c038a7b61a6a48447481d3982ff9fa1da55f3c` used runtime
`8f8f63f3250858667dfe460bdd22295c279f2625`, wheel
`53444f7f87141e7107fe9f4daf4a6095e04c9d69d007cd218634fd2fff50f99f`,
and the same pinned Hermes compatibility checkout and local Qwen3.5:9b
sampling profile as v3. The v4 protocol prospectively adds verification of
the installed write-followup tool descriptions; lifecycle criteria remain
unchanged. Live host configuration and memory stores were unchanged.

| Attempt | Passed phases | First failed phase | Observed failure |
| --- | --- | --- | --- |
| 1 | Teach, restarted recall, correction, restarted corrected recall, forget, source replay | Recall after forgetting | Newly captured assistant narration retained the deleted value. Retrieval returned it; later inspect/forget calls on deleted IDs returned HTTP 500. |
| 2 | Teach | First restarted recall | Bridge stdout contained a locale-encoded byte that could not be decoded as UTF-8. |
| 3 | Teach, first restarted recall | Correction | The model retrieved and inspected the target, then announced a correction without calling `mnemos_correct`. The target remained version 1. |

All three teach phases made the required native `mnemos_remember` and
`mnemos_job_status` calls. That is bounded progress from v3, whose three teach
phases omitted the status call. It does not establish reliable full lifecycle
behavior or isolate the causal effect of the guidance change.

Attempt 1 demonstrates the distinction between suppressing known source
replay and preventing newly captured narration from repeating a forgotten
fact. The original replay gate passed; the final recall still failed. Its
forget-phase assistant text was captured after deletion and included the
corrected label and old fact IDs. Treat this as a correctness gap, not a
successful forget demonstration. The later missing-fact HTTP errors are a
separate error-contract gap.

Attempt 2 reported `'utf-8' codec can't decode byte 0x97 in position 907`.
The bridge client requires UTF-8 while the server subprocess printed JSON
through the platform text encoding. Subsequent regression work must reproduce
this boundary directly; a different model response is not a repair.

Attempt 3 ended after `mnemos_retrieve` and `mnemos_inspect`. Its final answer
said it was correcting the fact, but the trace contains no correction call.
The evaluator correctly rejected both correction and same-ID version-two
gates. No later lifecycle phases ran in attempts 2 or 3.

The cohort completed with zero of three full passes. All host processes
terminated and cleanup was checked before subsequent scheduled attempts.
The three-host continuity prerequisite remains unmet. These observations use
the pinned server sampling profile without conflicting request overrides;
they do not expose internal sampler behavior.

`cohort.json` retains every outcome. `evidence-index.json` binds 137 files,
including executed helpers and protocols, request/response traces, selected
synthetic state, and logs. This README is interpretive and is outside that
raw evidence index. Databases, runtime access descriptors, credentials, and
live user content are excluded. Later fixes cannot change these results.
