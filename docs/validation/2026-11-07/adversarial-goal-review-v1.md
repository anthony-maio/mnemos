# Adversarial review of the improvement plan

Reviewed against local source checkpoint `8cf31899` on November 7, 2026.
This is a read-only review of source, plans, saved evidence and delivery
checkouts. It does not add test/model results or verify remote publication.

## Verdict

The work is credible, but the plan has become too focused on experiments and
verification cycles. Useful fixes exist. There is still no single current
configuration ready for daily use. Finish and audit the frozen named-support
run, then stop launching codec variants until the delivery and pilot plan is
concrete.

## Priority findings

1. **Quality remains unresolved in the product path.** The fresh retrieval
   evaluation returned a correct empty answer in zero of 510 observations with
   eligible facts. Retained source records establish that evidence exists;
   they do not establish that it answers the query. Experimental support
   selectors are not integrated into this path. The next quality milestone
   needs an actual retrieval-plus-support workflow, including unsupported
   answers and provider failures, followed by fresh end-to-end acceptance.
   [Evaluation](fresh-accepted-v1.md) and
   [service implementation](../../../mnemos/memory_service.py).

2. **Converge one candidate.** The installed product is `f5a944ea`, native Codex
   acceptance belongs to `c80eb051`, the recorded pilot configuration uses
   `6d0e5ba6`, and the prepared delivery patch targets `0456cb5d`. The reports
   disclose this honestly, but every additional product change creates another
   reconciliation cycle. Freeze a candidate, update its delivery and pilot
   drafts, and validate the selected host under normal trust settings.
   [Current plan](../../plans/2026-11-06-mnemos-improvements.md) and
   [delivery bundle](delivery-review-v2.md).

3. **Restore the original measurement distinction prospectively.** The
   original requirement is 100 ms for hook durable capture and 500 ms for warm
   recall. Applying 100 ms to the whole prompt command, including retrieval,
   creates a stricter combined requirement. Preserve every prior measurement
   and label; separately measure durable acknowledgment, retrieval and complete
   command time in a new protocol. Stop still fails the whole-command bound,
   and direct acknowledgment timing is unproven. This is not a readiness pass.
   [Original requirement, line 1017](../../plans/2026-09-16-functional-local-memory.md)
   and [complete comparison](../../plans/2026-11-07-fact-evidence-index-performance-v1.md).

4. **Deliver the storage repair independently.** The minimal backport changes
   one storage method and has regression coverage with a recorded 316-test
   pass. Recheck upstream state, complete package verification and prepare it
   for a concrete delivery decision. This repair should not wait for semantic
   abstention research or the larger pilot.
   [Backport preparation](../2026-11-06/storage-delivery-checkpoint.md).

5. **Make the pilot mode concrete.** The entry audit already excludes disabled
   automatic extraction correctly. Baseline answer support and normal host
   trust remain real gaps. Keep the current automatic-recall pilot inactive.
   A manual MCP workflow with explicit writes and evidence inspection could
   be proposed as an earlier learning pilot, but it requires a prospective
   scope decision, exact configuration and rollback. It must not silently
   waive existing requirements or count as completion of the full goal.
   [Pilot entry audit](pilot-entry-audit-v1.md).

## Evidence that holds up

The graph transaction repair has stale-client and rollback tests. One complete
native Codex lifecycle passed on its named artifact. The evidence index passed
isolated installation and a recorded 3,674-test regression. Its comparison
preserved all 160 output pairs and 208 durable captures. These are bounded,
useful results, not proof of semantic answer quality or completed daily use.

## Revised order

1. Complete and audit the current frozen run, then end this codec series.
2. Select one candidate and define the original timing contracts separately.
3. Refresh the small backport and current-candidate delivery packages.
4. Prepare a specific pilot mode, host validation, exact changes and rollback.
5. Give semantic quality one bounded development plan with an explicit stopping
   rule, then run fresh end-to-end acceptance. Preserve all six original goals.

## Follow-through at review time

The original timing wording was checked directly at line 1017. The current
named-support run remains frozen at `8cf31899`, session 49724. Its results must
be audited before changing tracked files or the source checkpoint. Further
codec variants are deferred in favor of candidate convergence and a bounded
end-to-end quality plan. Live configuration and publication remain unchanged.

## Result of the frozen run and response

The run completed all 272 observations and passed its independent audit at
`8cf31899`. The named format produced 41 unsupported selections against 14 for
the baseline and still failed the latency target. The codec series is closed.
The [convergence plan](../../plans/2026-11-07-candidate-convergence.md) freezes
product `f5a944ea` for delivery preparation, separates the minimal backport,
sets a two-run limit for one end-to-end quality increment and preserves the
original pilot and timing requirements. It does not declare those gates passed.

## Follow-up review of the revised plan

The read-only follow-up found no material findings. It confirmed that the codec
series is closed, the next quality increment has a two-run limit, prior timing
FAIL labels remain, and artifact validation, pilot activation and publication
stay separate. It did not rerun tests or independently audit the backport build.

One boundary remains explicit: results from the experimental support adapter
cannot qualify the frozen pilot's baseline retrieval configuration. Any later
pilot change needs its own exact artifact/configuration binding and acceptance.
