# Durability and history development comparison: v3 preparation

This supersedes the preserved, unexecuted v2 preparation. The first durability
run lost an independent project marker from both mixed-content events and
abstained on a useful observed completion timestamp. The v3 clause adds
assertion-level separation and single-event historical retention guidance.
No production prompt or evaluation criterion changed.

There are 46 reused cases, two arms and one repetition: 92 planned calls.
The internal runner arm label remains durability_v2; the exact v3 candidate
text is bound by inputs.json and preparation.json, not inferred from that label.
All original rubrics and pre-execution ambiguity reviews are preserved.
The 20 earlier history cases retain their unrelated existing facts; the 26
durability cases retain their empty existing-fact context. Source cc940ac is
verified, and sleep.py bytes match both prior experiment snapshots.

Offline preflight passed 92 case/arm schema, scope, prompt and options checks
with zero model generation calls. One stochastic development repetition cannot
establish reliability, unseen accuracy, service recovery or production adoption.
Review required meaning, forbidden meaning, attribution, uncertainty, conditions
and omissions across all cases. Keep semantic outcomes separate from runtime
validity. V1 terminal failures remain evidence rather than being overwritten.

Launch only after v1 is terminal, archived and reviewed. This archive contains
preparation only; no model result or started process is claimed here.
