# Relation inventory v2 execution result

The candidate failed its development screen and is closed. The run stopped at the first uncertain completion, a source request that exceeded the fixed 15-second deadline at 15,131.96 ms. It made 15 generation requests: 12 source attempts and three query interpretations. No retry, policy change or deadline extension occurred. The reuse pass never started.

Ten source attempts produced validated inventories. One completed reply was rejected after 14,659.77 ms because three purported quotations were absent from its source; the final request timed out without a complete reply. All six query observations were unavailable, and 130 scheduled query observations were unattempted. There were no selected facts, so precision is undefined. The fixed two-pass denominators receive zero retention credit out of 80 expected fact observations and zero successful-abstention credit out of 56 no-answer observations. This incomplete run is not an estimate of semantic accuracy over the full cohort.

## Completed-output review

All ten validated source inventories were reviewed against the frozen source targets. They contain four distinct inventory profiles; repeated profiles were compared for exact equality before applying the same semantic judgment. Eight whole-fact support assessments matched the targets. Two remained unresolved despite faithfully attributed, unverified report facts matching their own sources. Five inventories omitted governing prerequisites from the structured condition list while retaining the wording in the value text. All ten left temporal meaning unresolved. These counts describe the completed prefix, not all 127 planned sources.

All three completed query replies used claim index one for the second source, although its only claim had source-local index zero. Exact-cohort validation rejected each reply. The request serialized nested claim arrays without explicit claim-index fields, leaving a protocol weakness: the model supplied a global-looking ordinal instead of the expected source-local index. This does not justify repairing or regrading these replies. A future interface should transmit explicit stable claim references.

The completed conditional-source reply also supplied paraphrased or nonexistent quotation strings: lowercase `approval`, `If approved, Tern will raise its worker limit to 24`, and `status`. The exact quotation boundary correctly rejected them. Changing to fuzzy matching would weaken provenance and is not supported by this result.

The temporal and condition findings remain separate from the index problem. Correcting query references alone would not establish source completeness, whole-fact support, applicability or latency. No full retrieval timing or warm-cache benefit was established.

## Evidence and cleanup

Wrapper session 48682 ended with exit code zero after handling the prescribed stop. The model was unloaded, parent/native processes were absent, both ports were closed, and runtime/prepared bindings were unchanged. The owned server returned code one after the explicit process-tree stop.

The independent audit verified the complete event journal, terminal summary, scheduled denominators and cleanup receipt. Its machine-readable result is `relation-inventory-execution-v2.json`. The 17-member evidence archive is `.venv/relation-inventory-execution-v2-evidence.zip`, SHA-256 `7594050a76b9859068f320ddeead157515c9488482fc6f935c22b4c8662fa66f`. Completed-output review is retained separately at `.venv/relation-inventory-completed-review-v2.json`, SHA-256 `21d611ff6accc07a7045cc03a437a61d114e47d0bf6b6857e547bdf2a46adb11`. It binds each reviewed observation by its journal digest and leaves original scores unchanged.

Do not rerun this candidate or reopen its deadline. A subsequent candidate must address the source annotation and interface limitations prospectively, with new frozen policies and evidence. Original quality, latency, fresh acceptance, daily-use and delivery obligations remain open.
