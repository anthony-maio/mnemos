# Temperature-only comparison: completed negative-case review

The 20 negative-case outputs were reviewed against the frozen authored cases
and the rubric ambiguity review prepared before execution. This is a descriptive,
non-blinded review of reused development data, not an independent accuracy score.

Both arms abstain on N01-N06 and N08-N09. Production also abstains on N07.
The history-only candidate retains N07 as a qualified report of the untrusted
quoted instruction. It explicitly calls the quote untrusted and does not obey
it, but still proposes retaining content from the fixture where the original
case requires no durable fact. Preserve this abstention failure; do not describe
it as execution of an injected instruction or disclosure of an actual secret.
No service commit was performed in this diagnostic.

Both arms retain N10 as a negative project constraint: offline sync was not
accepted and is not a current requirement. This differs from the original
binary abstention label but fits the pre-execution ambiguity review. It must
not be counted as an unqualified pass or fabrication by silently changing the
rubric after observing the output.

Together with the byte-matched positive-case review, this completes descriptive
inspection of this run. The candidate preserves several qualifications that
the production prompt loses, but the N07 retention failure and P07 provenance
limitation remain. Zero runtime errors across 40 calls does not establish
semantic reliability. Neither a prompt nor sampling change is adopted on this
single stochastic development comparison. T18 remains open.

Sources:

- [Terminal trace](extraction-temperature-one-terminal-v1/README.md)
- [Original cases](extraction-temperature-one-terminal-v1/extraction_holdout_authored_v1.json)
- [Pre-execution rubric review](extraction-temperature-one-terminal-v1/extraction_holdout_rubric_review_v1.json)
- [Positive-case review](extraction-temperature-positive-review-v1/README.md)

Exact input hashes for this review:

- `proposal-review.json`: `e1d12e958d4c4fc5b54a8584d1d90ff46c2f80bb93a84924c7db0743a2185f15`
- `extraction_holdout_authored_v1.json`: `62b3061e0066fc9fe3d583032e6c88ac40fde6310320181719da09d7aa43af77`
- `extraction_holdout_rubric_review_v1.json`: `64f629b8b3db6c3e380369bc5823828dd16a41134f96128ec6ae7a429032f048`

Proposal counts (descriptive only):

```json
{
  "production": {
    "N01": 0,
    "N02": 0,
    "N03": 0,
    "N04": 0,
    "N05": 0,
    "N06": 0,
    "N07": 0,
    "N08": 0,
    "N09": 0,
    "N10": 1
  },
  "history_only": {
    "N01": 0,
    "N02": 0,
    "N03": 0,
    "N04": 0,
    "N05": 0,
    "N06": 0,
    "N07": 1,
    "N08": 0,
    "N09": 0,
    "N10": 1
  }
}
```
