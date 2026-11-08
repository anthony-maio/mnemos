# Portable GQA retained-model numerical failure

The candidate failed its frozen full-model numerical gate and stopped before timing. No capacity, semantic-quality or training result follows. The earlier 876.17 ms configuration remains a separate failed result.

Independent review verified the freeze and all bound files, agreement between result and parity records, all 28 dispatch observations, and absence of timing samples. This is failure against the declared numerical contract; it does not establish semantic degradation.

Runner/protocol commit: `ed29fa29`. Private execution directory: `.venv/portable-gqa-capacity-v1`. The process executed once and exited 1. Its exclusive start marker remains. Preparation copied the original six request/frame files byte-for-byte; all frozen source, checkpoint, runtime-source and input bindings were verified again after failure.

| Workload | Result |
| --- | --- |
| Empty | Passed; no logits, hidden states or attention observations |
| One-short, 98 tokens | Failed full hidden-state comparison |
| Remaining four workloads | Unrun because the numerical gate failed |
| Warmups and measured repetitions | Unrun; result samples empty and no samples.jsonl |

For one-short, all compared values were finite. The one random-head target logit differed by 0.0013427734375 and passed the declared atol=0.02, rtol=0.02 check. Of 100,352 final hidden-state elements, 28 exceeded that same elementwise tolerance, a fraction of 0.00027901787. Maximum absolute difference across all hidden elements was 0.140625. That maximum alone is not the failure criterion; the count beyond atol + rtol * abs(reference) is. Tolerance is unchanged, and a passing random-head logit cannot override the failed full-hidden gate.

The untimed candidate pass recorded 28 attention observations. Each selected repeated K/V representation with final query/key/value shape [1,16,98,128], float16 on cuda:0, no mask, causal=true, dropout=0, gradients disabled and enable_gqa=false. The private chooser reported EFFICIENT (2) at every layer. These are selector observations from the untimed pass, not a trace of timed kernels. No timed kernels were run. All four backend enable flags remained true. Checkpoint loading reported no missing, unexpected, mismatched or errored keys.

The review caught an interruption-handling gap before freezing. The runner now records KeyboardInterrupt and SystemExit as terminal failures, preserving active sample identity and invalidating any provisional pass. Four new failure/restart checks passed. The adapter's eight existing numerical tests also passed, illustrating why small/tiny-model parity could not substitute for this retained-checkpoint gate.

## Evidence bindings

- Freeze: `698fec02646c51047fe10a301cc73723d8c21e63491fdf6a7804404b7ea2a660`.
- Result: `5eb8ff27a003da125ebd5d99d85af421f95fcc2a9cdd1661e04acaec3132fa3e`.
- Parity: `352e0f37e4797cc42de960357ebcb9f477d092b001e46d2242501d2d180aa6d8`.
- Execution log: `6ad3932202f918e7da1af263d2d43a863c5c24695ceb0b176c116a816edf1fb1`.
- Private archive `.venv/portable-gqa-capacity-evidence-v1.zip`: `c7880b2c7927fddc196acd30660830a45eecbde8da800ad29cd84eb1f3a91c60`, 14 members, every manifest member hash verified after readback.

The archive retains requests, frames, freeze, start marker, logs, result, parity observations, runner, adapter, head, serializer, protocol, interruption tests and hash manifest. Checkpoint and installed runtime sources remain bound by the freeze rather than duplicated in the archive.

## Decision

Close this candidate. Do not relax tolerance, substitute logits-only parity, shorten the workload, or run timing after this failure. No trained support model or passing computational candidate exists. Further semantic-path work requires a new substantive computation/design decision, with a separately justified numerical and complete-latency contract. Existing data and all failures remain available; no fresh acceptance data was consumed. Capture latency, original native acceptance, actual pilot use and delivery remain outstanding.
