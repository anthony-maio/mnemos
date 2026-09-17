# Disposable SQLite drive comparison

On September 17, identical one-row SQLite updates under WAL and synchronous
FULL had median commit times of 2.040 ms on C: and 70.805 ms on D:. SQL execution
medians were 18.6 and 14.8 microseconds respectively. Each drive had three
warmups and twenty measured updates, alternating drive order each round.
After close and reopen, each database retained its one row and counter 23.

This comparison supports using fresh C: fixtures to investigate the installed
continuity timeouts. It does not prove storage was the only cause of earlier
failures, establish a product speedup, or close the 10,000-fact recall gate.
The shared machine was not isolated from other work. The original failures
and D: measurements remain valid observations of those runs.

The report preserves all samples, settings, paths, command, Python version,
disk inventory, and executed helper hash. Raw files are copied unchanged.
No live database, host configuration, or durability setting was changed.
