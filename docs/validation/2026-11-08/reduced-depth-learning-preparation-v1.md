# Learning preparation and coverage decision

The training mass allocator is implemented, with 22 loss/data-integrity tests
passing. It includes all 282 training targets in 67 queries and uses connected
families, classes, negative reasons and queries to control imbalance. Tests
verify mass conservation, family/class/reason/query balance, complete target
accounting, order invariance, no input mutation, reviewed-label enforcement and
independence from reserved label decisions. Formatting and diff checks pass.

The private `.venv/reduced-depth-learning-preparation-v1/weights.json` binds
the original inputs, labels and implementation, with every target weight. Its
SHA256 is `1af1b3a54f07405306c0a63920396cbec5b9b0ba1bcd84d921115dc038c49bc1`.
JUnit is `.venv/reduced-depth-loss-v1.xml`. These are preparation artifacts,
not a trained model or proof that labels are semantically sufficient.

The [recipe proposal](../../plans/2026-11-08-reduced-depth-learning-v1.md)
defines the objective and proposes rank-8 q/v adaptation, 50 full-corpus updates,
one final checkpoint and a fixed threshold. Runner/scaler/baseline binding and
adapter implementation are still outstanding. No model training has started.

Independent coverage review found corpus-v2 insufficient under its existing
sparse-category gate. The one unknown-signature condition contrast and two
attribution negatives from one retraction family cannot establish the intended
breadth. Five critical opportunities cover only two facts. Next, add genuinely
discriminative quorum/veto and negation-scope attribution constructions with
positive counterparts, preserve corpus-v2, and review the new labels and lineage
before training. The remedy is better contrasts, not arbitrary counts, relabeling
known cases as fresh acceptance or changing the original quality thresholds.
