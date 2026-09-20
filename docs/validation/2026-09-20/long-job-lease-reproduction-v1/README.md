# Active work can outlive the fixed maintenance lease

Reproduced on installed product d937892 with a controlled clock. One job is
claimed for 300 seconds, its provider-work stand-in succeeds at second 301,
and durable completion rejects the expired lease. There is no competing
worker during processing. The scheduler completes zero jobs; a subsequent
claim consumes attempt two for the same work. No actual model call ran.

This is a scheduler lifetime gap, not an argument for accepting expired tokens.
One all-module extraction can make multiple configured provider calls, so a
fixed lease can be shorter than healthy processing. The current scheduler
has no renewal. A source audit independently confirmed the missing mechanism.

The repair must renew only an unexpired lease with the same owner and token.
Renewal should extend to at least now plus the lease duration without shortening
an existing later deadline. It must not accumulate another full duration on
every heartbeat, which would delay crash recovery unnecessarily. The scheduler
must stop and await renewal/processing tasks on completion, failure or cancel,
and must not let a stale worker mutate a reclaimed job.

Required checks include work beyond the original deadline, another owner being
unable to claim a renewed lease, stale-token rejection after reclaim, expiry
without resurrection, cancellation leaving recoverable work, and failure or
completion racing with renewal. This archive records the unresolved baseline;
it does not claim a production repair or a real-model incident.
