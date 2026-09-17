# Preference extraction development check

Eight cases were frozen before one provider call per case using Qwen3.5:9b,
thinking disabled, temperature 0, and a 512-token limit. The actual
`OllamaProvider` and `SleepDaemon.propose_facts` path produced eight
runtime-valid responses. Exact text-match proxy: 4/8, unchanged.

Post-run qualitative review against the authored semantic criteria passes
seven cases: two stated preferences, an explicit correction, and abstention
on a question, assistant claim, fictional quotation, and hypothetical. The
valid preference and correction paraphrases do not change their failed exact
proxy scores. This small development check is not held-out accuracy or human
acceptance evidence.

The undeployed Redis requirement fails. The model returns the stored SQLite
statement as a conflict, citing the new source even though it does not support
that statement. It also omits the requirement. A valid source ID and response
schema do not establish factual support; this extraction limitation remains
open. The probe proposes facts and does not test a durable service commit.

The dirty candidate's base is 64471f5. Its source snapshots include the explicit
thinking control and preference/requirement prompt clarification integrated
in d94ded6. The manifest, original runner, all source bytes, raw responses,
and separate qualitative review are retained. Aggregate source and individual
file hashes were reproduced before archival. No labels or thresholds changed.
