# Identity coverage v10 preparation

The opt-in identity-coverage treatment and benchmark capture/grading path are implemented. The check includes subject, property and applicability information. It can turn either a premature match or a premature unrelated judgment into insufficient identity. It never turns an unrelated or uncertain pair into a complete match. The earlier qualifier-only treatment and all default comparator behavior remain available unchanged.

The benchmark retains both original referent responses and added coverage responses. Relation grading validates both complete response sets, reconstructs the effective relation, and assigns no relation credit for missing, extra, malformed or failed coverage. Final operation grading still uses the original labels and source/target snapshot. All targets participate in the existing ambiguity decision.

Validation passed 97 focused tests, including 15 new checks, plus strict type checking on four runtime/benchmark modules. The preflight generated 396 planned requests across 34 literal pairs and two layers. It replayed all 198 v8 requests with original responses and recovered unchanged runtime results. Each layer's prompt and schema were invariant across storage IDs, target order and distractors. No new model calls occurred.

The prospective protocol keeps the original 72-view and 198-relation gate, model package, sampling controls and resource limits. Runtime model/tokenizer validation and native token counting remain required before inference. Structural reachability and contract tests do not predict semantic accuracy. In particular, the new check may mistakenly suppress correct comparisons; the study must count lost corrections, wrong additions, incorrect reviews and unsafe targets.

The candidate is isolated in the improvement worktree. Canonical delivery, installed packages, live memory and pilot configuration are unchanged. Run the frozen development study next, preserve every observation, and independently audit both response layers before deciding whether it merits further confirmation.
