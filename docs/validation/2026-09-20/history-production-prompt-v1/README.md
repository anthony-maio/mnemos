# Unchanged production prompt with managed structured reasoning

All 36 calls completed and passed runtime response validation on source
05c71d2 using the managed production provider, local Qwen3.5:9b, thinking
enabled, temperature zero, an 8,192-token budget, and a 180-second timeout.
The extraction prompt was unchanged. Twelve previously authored development
cases each ran three times. No facts were committed.

Manual inspection finds abstention on the historical approval and unverified
relay in all three repetitions. The fiction input yields a team-policy
prohibition in all three repetitions, failing its frozen zero-fact control.
The question, hypothetical, and unaccepted assistant suggestion abstain.
All final proposals are copied into proposal-review.json for direct review.
Runtime validity does not establish semantic correctness.

The earlier history-clause chat comparison used an experimental provider and
an older source snapshot. Do not present their difference as a one-factor
controlled prompt experiment. The separately prepared authored holdout uses
the same managed provider and source in both prompt arms. The candidate prompt
has not been adopted. Service lifecycle and broader quality remain open.
