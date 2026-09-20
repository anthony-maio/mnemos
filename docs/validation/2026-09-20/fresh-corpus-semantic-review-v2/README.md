# Fresh corpus v2: preparation defects

The [preserved v2 draft](../fresh-corpus-draft-v2/README.md) is not ready for
acceptance evaluation. This review made zero evaluated-model calls and changed
no production prompts, thresholds or historical results. The accompanying
audit binds the exact input hash and checks every heldout correction, supported
fact pool and state-pair declaration. V1 and V2 remain preserved.

## Corrections do not replace a substantive decision

Seven of 20 corrections only insert "now" into the original assertion.
The other 13 append "The current review window is Wednesday." without changing
the original assertion. Marking the original decision superseded is therefore
not a useful test of a changed decision. In those 13 cases the old text is also
a substring of the valid new text, making stale-substring detection ambiguous.
Repair the scenarios; do not weaken the existing scorer to force a pass.

All 20 correction targets identify a generic message saying the earlier rule
was provisional. The actual assertion appears in a different earlier capture.
The old fact's evidence points to the generic message, so the runtime may have
no accepted fact from the designated source to correct. This is a fixture
provenance defect, not evidence of a new product failure. Use the actual source
assertion and make its relationship to the correction explicit.

The replacement body itself is authoritative input: ollama_adapter.py passes
replacement_content as the correction payload's content. A generic visible
correction instruction alone therefore does not prove that the replacement is
unsupported. This limits an earlier review concern about that representation.

## State and ranking coverage is inadequate

All 80 pairs use neutral paraphrases and label semantic_state as "supported".
They do not hold a question constant while changing affective context. All 80
also declare the same fact as direct vector seed and graph target, which cannot
establish associative retrieval beyond direct retrieval.

Every heldout session has exactly one current supported fact. Top-five success
in that pool cannot demonstrate ranking among competing relevant and irrelevant
memories. Add meaningful competing facts and precise questions, and document
which cases actually exercise ranking. Do not infer realistic context pressure
from nine short events labeled long_session.

## Coverage and limits

The draft does contain 150 ordinary no-answer queries and 50 foreign-scope
canaries. A separate semantic review reported no obvious same-session answer
leakage, but its lexical screening is not proof of unanswerability. Review those
labels again when competing facts are added.

Development and heldout scenarios reuse semantic templates. Number changes and
zero exact overlap with prior corpora do not establish disjoint semantic
coverage. Conversely, an unexecuted authoring revision is not evidence that an
evaluated model has seen the dataset. Record authoring dependencies and separate
development families before freezing; no arbitrary 200-family minimum is added.

Next: author v3 with meaningful correction values, correct source links,
competing fact pools, matched state variations and separated development
families. Review it before model execution. Structural validity remains a
necessary check, not semantic acceptance.
