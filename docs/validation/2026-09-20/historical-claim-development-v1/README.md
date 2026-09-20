# Historical claim extraction development comparison

Twelve new development cases compare the production extraction prompt with
one added attribution/history clarification. Each arm runs three repetitions
through the actual `SleepDaemon.propose_facts` and Ollama provider from verified
source `ac1a838`. Cases, semantic review criteria, arms, and request settings
were frozen before each run. There are 72 runtime-valid responses per model,
144 total, with no request or schema errors and no fact commits.

Qwen3.5:9b completed first. Because its added clarification did not recover
historical claims, a second comparison used the already-installed
Qwen2.5:14b. Cases, criteria, both prompts, temperature zero, 512-token limit,
and three repetitions remained unchanged. Thinking was disabled for
Qwen3.5 and omitted for Qwen2.5, whose reported capabilities lack thinking.
This second model choice was made after seeing the first result; it is a
development comparison, not an independently held-out model selection test.

| Observed behavior | Qwen3.5:9b | Qwen2.5:14b |
| --- | --- | --- |
| Current policy and preference | Both arms retain both cases, all repetitions | Both arms retain both cases, all repetitions |
| Five historical/attributed cases | Both arms abstain in all 15 observations | Both arms produce proposals in all 15 observations |
| Future requirement | Both arms abstain in all three observations | Both arms retain the requirement and incomplete implementation status |
| Fiction, question, hypothetical, assistant suggestion | Both arms abstain on all four controls | Both arms abstain on all four controls |

Nonempty proposals are not automatically correct. The Qwen2.5 baseline turns
"The earlier project note said that archival exports used Zstandard" into
"Archival exports use Zstandard" in all three repetitions, losing attribution
and historical tense. Its candidate preserves the earlier-note qualification
in all three. The candidate also explicitly preserves the user's statement
that Mira's port report was unverified; the baseline names Mira but omits that
qualification. Other observed candidate outputs retain obsolete status,
discontinued Trello use, and the future requirement without claiming work
was completed. These are qualitative observations against the frozen authored
criteria, not independent human annotation or a general accuracy score.

The clarification alone did not repair Qwen3.5. It remains a development
candidate: no production prompt or live model selection changed during this
comparison. A stronger next check would exercise accepted facts, correction,
forgetting, and retrieval through the service, preserving all qualifications
and testing rejection controls on data separate from this development set.

Each model directory preserves exact inputs, preparation bindings, runner,
raw prompts/responses, provider metadata, terminal state, and a residency
sample. `summary.json` groups distinct proposal texts without assigning an
overall semantic score. The collector verifies identical prompts except for
the frozen candidate clause, identical request schemas, all 72 case/arm/
repetition identities, and original source hashes. Indexed source files come
from the verified snapshot. The two GPU samples report a 16,384-token context
and positive residency; they do not prove GPU behavior for every request.
Cold loading is included, and no ordinary recall latency is inferred.

The original 4,200-observation matrix and its 50 incomplete cases remain
unchanged. This archive does not promote that run to acceptance.
