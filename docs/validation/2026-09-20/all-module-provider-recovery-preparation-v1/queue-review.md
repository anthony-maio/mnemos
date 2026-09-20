# Recovery queue review

Read-only review found no path that would falsely report a successful recovery
from the frozen runner's failed lifecycle assertions. It identified one queue
limitation: the first launcher checked the terminal result and hashes without
checking the bound process exit code.

The original queue process was stopped before the comparison completed and
before any recovery result existed. The comparison process was left running.
Queue v2 holds synchronization and query access to the same prerequisite
process and requires exit code zero, terminal completed state, 40 observations,
and matching preparation/output hashes before launching recovery once. The
original queue source remains preserved; queue-v2.py.txt is the active version.

The Windows exit-code primitive was checked with two disposable processes:
exit zero was accepted and exit seven rejected. This is an offline queue check,
not model or recovery evidence. The frozen recovery runner and database inputs
were not changed. Marker checks identify the synthetic lifecycle target; they
do not establish exact semantic fidelity of the entire extracted fact.
