# Claude capture association diagnostic

An offline fake-client replay against the installed 6600dc2 adapter reproduced
the v13 capture warnings. A prompt hook without prompt_id created a provisional
user record; Stop could not safely associate it with the transcript user row
and reported ambiguous_transcript_association. Assistant capture succeeded.

Supplying the final transcript prompt ID as a control produced five captures
and successful hook responses, compared with six captures in the unbound case.
The extra unbound record is provisional and is not evidence of duplicate
durable facts. This control used a completed transcript, so it does not prove
that the identifier existed when the actual prompt hook ran. A separate native
hook-time observation is required before designing an association repair.

The report and helper are preserved byte for byte. The adapter hash is verified
at archival time; the original report does not bind a helper hash at execution.
No new native host call or model inference was performed by this diagnostic.
