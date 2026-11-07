# Installed performance check: Windows service ownership

V3 retains the [original protocol](2026-11-07-installed-performance-v1.md) and
the [configured-store identity repair](2026-11-07-installed-performance-v2.md).
The wheel, model, data content, sample counts, warmups, complete hook-command
timers and 500 ms/100 ms limits are unchanged. Use a new private runtime;
preserve both failed attempts and their archives.

The Windows virtual environment launches a child interpreter. V2 required the
runtime descriptor to name the launcher's PID and therefore stopped before any
timed requests. A model-free live probe verified that the interpreter has a
different PID and the expected launcher as its direct parent.

V3 must inspect the claimed owner through the operating system while the
launcher is live. Accept either the directly launched interpreter or its direct
child, with the expected parent, exact interpreter executable and exact private
`service-entry.py` command. Parse Windows command-line arguments rather than
matching a path substring. Record the process creation time and the ownership
proof. A model-free probe must accept the real child and reject an unrelated
parent, command or executable before inference begins.

Register the verified interpreter separately from the launcher. Before sending
authenticated shutdown, confirm the same process identity still exists. Then
wait for the launcher and verify that the interpreter and runtime descriptor
are gone. A fallback termination makes cleanup unsuccessful for acceptance;
never stop a process merely because its PID appears in a descriptor.

Ownership checks happen during setup and cleanup, outside warm retrieval and
hook timers. Cold-service startup is an observed upper bound that includes the
ownership check; record that check's duration separately. Do not subtract
interpreter startup from complete hook-command measurements.

This repairs benchmark process supervision. It changes neither product source
nor the timed request paths. V3 must retain all planned observations and any
failures. No native-host, quality, daily-use or release acceptance follows from
this amendment.
