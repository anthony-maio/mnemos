# Answer support and context efficiency

Status: development protocol. No new model experiment has run under this
protocol. Historical scores and release gates remain unchanged.

## Question

Can a bounded support check reduce unsupported recall without losing useful
facts or exceeding the local latency and context budgets?

The October recovery report identifies two different problems: some returned
texts contain the expected answer with additional surrounding text, and other
texts are merely related to a question with no answer in memory. The first is
partly a boundary/scoring problem. The second requires a support decision.
Neither is evidence that graph or state boosts bypass the semantic floor.

## Contracts

1. Retrieval candidates are immutable accepted facts with fact ID, version,
   scope, assertion status, source evidence and an embedding identity. Capture
   and extraction are disabled during the retrieval experiment. Every arm sees
   the same candidate snapshot, query, scope and budget.
2. A supported answer must identify a current candidate and the evidence that
   supports the requested relationship, with its subject, time, attribution,
   negation and uncertainty intact. Shared vocabulary or the presence of both
   entities does not establish a relationship between them.
3. The support check returns a typed disposition and selected evidence
   identities/spans. It cannot generate memory text, change facts, accept
   instructions embedded in a fact, widen scope, or promote captured evidence
   into accepted memory. Quoted instructions remain quoted data.
4. Related, unsupported and unresolved candidates are distinct outcomes.
   Unresolved conflicts may be shown as conflicts, never silently resolved by
   ranking. A negative or uncertain statement can support a question when its
   qualification answers that question; uncertainty is not a blanket exclusion.
5. Provider failure is an explicit unavailable/degraded result, not a confirmed
   no-answer judgment. An optional support mode may retain clearly labeled
   search results; a required support mode must not describe unverified results
   as supported. No automatic cloud fallback is permitted.
6. Any provider await requires revalidating fact versions, scope and evidence
   before emitting context. Correction or forgetting during the request must
   invalidate the assessed candidate. Access metadata updates remain separate
   from changes to fact content.
7. Context packing uses complete qualified records. Concise spans must preserve
   necessary conditions and attribution and remain traceable to the immutable
   fact and source. If that cannot be established, retain the full fact or omit
   it for lack of budget. Never clip a record midway.
8. An optional host tokenizer measures the final concatenated context, including
   IDs and qualifiers. The byte bound remains the fallback when no compatible
   tokenizer is supplied. Reports distinguish actual tokenizer counts from an
   upper bound and identify the tokenizer and revision. Invalid counters cannot
   produce a successful zero-cost context.

## Development sequence

- Reuse the accepted-snapshot mechanism in
  `benchmarks/local_memory/retrieval_comparison.py`. Add a separate experiment
  namespace and fixtures; do not rewrite the old protocol or its artifacts.
- Establish a scoped semantic/lexical baseline and record candidate IDs, scores
  and source evidence before support decisions. Compare baseline retrieval and
  baseline plus support on that same snapshot. Compare full ranking separately
  so extraction, ranking and support are not combined into one treatment.
- Start with complete candidate facts as selectable evidence. Add shorter spans
  only after support behavior is measured, then compare packing independently.
- Use the existing typed-judge boundary for a local optional prototype. Pin the
  model, prompt, probability policy and request limits. Model confidence alone
  is not proof of support or calibration.
- Exercise missing entities, absent relationships between known entities,
  transient status, old/current facts, negation, attribution, quoted commands,
  conflicting facts, explicit uncertainty, paraphrases and identifiers/paths.
  Include straightforward answerable controls for every abstention category.
- Record development failures and diagnose whether a miss happened in candidate
  retrieval, support selection, evidence validation or context packing. Do not
  label all empty responses successful abstention.

## Prospective evaluation

Before running the evaluation, freeze a new corpus, labels, rubric, source
commit, snapshot hashes, model/tokenizer identities, options and all thresholds.
Use at least 200 new held-out session scenarios with at least 100 answerable
queries and 100 no-answer queries, distinct from development and the historical
corpora. Keep the existing critical-retention and extraction requirements in a
separate extraction evaluation. Do not claim blindness after inspecting or
tuning on evaluation failures; subsequent runs become regression results.

Run three repetitions. Report all attempted observations, failures, timeouts,
degraded responses and exclusions, with session-clustered paired differences
and proportion intervals. Provider failures count against operational success
and are never credited as correct abstention.

Report separately:

- Supported answer recall and unsupported-result rate on answerable questions.
- Correct abstention and false-answer rate on no-answer questions.
- Exact text match, literal answer containment and semantic support judgments.
  Containment alone is not a semantic pass.
- Conflict/uncertainty preservation, scope leakage and stale-version returns.
- Candidate retrieval recall, then support selection accuracy conditional on
  the required evidence being present. This reveals which stage needs repair.
- Context tokens, supported facts per token, omitted facts, model calls and
  warm/cold end-to-end latency on recorded hardware. A byte bound is not a
  measured tokenizer count.

The existing targets remain: at least 90% answerable Recall@5 and 90% correct
abstention, with zero scope leaks or stale/forgotten-fact resurrection in the
deterministic adversarial checks. Do not accept a support mode solely because
it abstains more: compare answerable recall and latency with the fixed baseline.
Default promotion also requires the applicable native-host and pilot evidence.
Neutral or negative results leave the feature experimental and remain recorded.

## Extraction evaluation

Evaluate event-to-fact extraction independently on new source events. Preserve
event identity, role, source spans and scope; grade supported-fact precision,
critical retention, temporary-text retention, revisions and forgetting. Feed
the same preaccepted facts to retrieval arms even when an extraction arm fails.
Report the extraction failure rather than silently removing its downstream
query from a denominator. A separate end-to-end result can then measure the
combined system without being mislabeled a ranking comparison.
