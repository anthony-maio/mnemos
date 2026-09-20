# Fixed-candidate graph-weight bound

This post-hoc diagnostic reads the frozen development report and changes no
settings or criteria. It examines repetition zero: five cases, two embedding
contracts and three graph arms. The original top-five result remains zero.

Holding each recorded candidate pool and activation energy fixed, the diagnostic
checks the linear score's endpoints and all target-versus-candidate crossings
inside the allowed graph-weight range [0, 1], plus intervening intervals. Ties
favor the target, making this an optimistic rank bound. The recorded score
arithmetic is checked at the original weight 0.1.

Summary: {"rows": 30, "target_missing": 15, "could_reach_top5_in_optimistic_bound": 3}.

This does not execute retrieval, enforce context budgets, test a chosen weight,
or estimate held-out quality. A possible top-five bound is a hypothesis, not a
successful result or recommendation to change the default. Missing candidates
cannot be recovered by score weights alone. The separate seed audit identifies
another coverage limitation. Original lifecycle failures remain preserved.
