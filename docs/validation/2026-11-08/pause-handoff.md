# Pause checkpoint

Paused at the user's request to evaluate the work so far. No further native
attempt, product change, package publication or live configuration change is
scheduled by this handoff. Resume only on the user's instruction.

## Current state

The work has produced source repairs and substantial validation infrastructure,
but the full six-priority goal is not achieved. Passing regression tests do not
establish semantic quality, latency, daily-use usefulness or release readiness.

- Graph persistence and storage repairs were prepared and merged in the earlier
  patch work. PR #9 remains separate from this broader development effort;
  recheck its remote state before any merge or release action.
- Answer-support/extraction development has recorded failures. The latest
  trained joint candidate did not meet the quality or merge checks. No fresh
  held-out result establishes the required quality gates.
- Extraction and retrieval have separate evaluation work and preserved results.
  The full quality and latency obligations remain open.
- Older native workflows passed against their own bound versions. The current
  v6 and v7 Claude attempts failed. V7 passed the first four phases and completed
  seven additional inspections, then timed out without deletion. Final source
  replay and restart recall were not performed. Cleanup succeeded.
- Context-budget/tokenizer support and review presentation improvements exist.
  The latest batch interface has no demonstrated native cost or latency benefit.
- Source, package, installed pilot, website and published release remain distinct
  states. The latest source changes are not installed or published. The original
  14-day pilot is not complete; optional pilot activation is not evidence of
  daily use or completion of that obligation. Desktop validation remains open.

## Saved work

- Delivery: `D:/Development/mnemos-delivery`, commit
  `a75cb609d0bcf8bd739d75a6b5c1f02f1181ee74`, pushed to
  `wt/backup-20261108-delivery`. Latest change labels captured evidence and
  assertion status in forget previews and binds those labels to the digest.
  All 104 surrounding tests passed; independent review found no blocker.
- Improvement worktree: `D:/Development/mnemos/mnemos/.worktrees/mnemos-improvements`,
  branch `wt/mnemos-improvements`, pushed to `wt/backup-20261108-improvements`.
  This checkpoint is committed there.
- Original dirty 0.7.0 checkout remains untouched. Its source backup is
  `wt/backup-20261108-local-070` at `cc5ecf681c261409aad30c99e4fa52a1c47786dd`.
  Sleep settings, episode identity and benchmark chronology were integrated.
  FastEmbed implementation comparison is complete; auto-priming, first-run UI
  and broader durable-pipeline comparisons remain open.
- Raw native evidence, environments, credentials and databases are not included
  in the remote source backup. Private v7 evidence is archived locally as
  `.venv/native-claude-batch-v7-evidence.zip`, hash
  `4b75550cdc32b15b7aa43499ccd6556bea192b7b31c9edfa38b589fe07145a93`.
  The native process session is terminal; its owned job reports zero survivors.

## Last completed measurement

The bound v7 trace reconstructs 19 preview candidates, 22 sources and seven
complete additional inspections. Results are saved in
`native-review-payload-v7-metrics.json`.

| Representation | Compact snapshot bytes | Paged inner bytes | Added bytes |
| --- | ---: | ---: | ---: |
| Preview | 43,797 | 51,615 | 7,818 |
| Inspection batch | 22,259 | 26,903 | 4,644 |

These are UTF-8 byte counts, not host tokens, cost or latency. The largest
individual inspection result envelope is 4,585 bytes, excluding fragment/page
metadata. This does not prove a complete rendered page fits the 12,000-byte cap.

The original private measurement helper used the wrong normalized position
field names, causing ordering checks to use the legacy fallback. It remains
preserved. The corrected `.venv/measure-native-review-v7-v2.py` uses `line` and
`invocation_line`; its new receipt passes reconstruction, digest and ordering
checks with unchanged byte totals. Its hash is bound in the committed metrics.
The helper ran offline and did not mutate the source trace or database.

If the user resumes this direction, whole-record packing within the existing
fragment format is a hypothesis to investigate offline. It must preserve exact
content, IDs, versions, source dependencies, ordering, complete inspection and
stale-preview checks, including deterministic layout/cursor compatibility.
No implementation or native attempt for that hypothesis has begun. The user
should first assess whether continuing this direction is the right use of time.
