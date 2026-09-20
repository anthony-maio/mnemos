# Hermes revision lifecycle: correction collision remains open

The exact three-record v32 outbox was delivered into a new private service
fixture using the byte-verified installed wheel from product 05c71d2.
Delivery, stopped-service buffering, restart and duplicate replay passed.
Maintenance completed both original-source jobs as captured evidence and
kept the derived summary ineligible for facts.

Explicit correction of "Use cedar export format." to the already captured
"Use birch export format." was accepted as a durable request. Its maintenance
job failed with RuntimeError and remained pending after one attempt. The old
target stayed at version 1, unchanged. The lifecycle verifier failed here;
forgetting and subsequent replay phases did not run.

Storage has an explicit expected-version canonical-collision guard. Its
contract and safe recovery behavior require review before changing it: an
implicit merge must not discard source history, versions or scope boundaries.
This record does not establish the exact exception message from the worker,
which stores only the exception class. No retries or live-store edits were
performed to make this attempt pass.

All modules were disabled, with an unavailable language model and simple
embeddings. This is a structural lifecycle failure, not an extraction-quality
measurement. Service descriptor cleanup passed. The copied database passes
integrity and foreign-key checks. Original v32/v33 evidence is untouched.
