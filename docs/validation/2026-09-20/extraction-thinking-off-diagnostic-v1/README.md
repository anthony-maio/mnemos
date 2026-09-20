# Thinking-disabled diagnostic on previously executed cases

All 40 planned calls are recorded: 20 synthetic cases, two prompt arms, one
repetition per case and arm. There are 40 runtime-valid responses and
0 recorded runtime errors. No failed call was selectively rerun. The
runtime summary separates the production and history-only arms; raw requests,
responses, failure records, final proposals, and frozen preparation are included.

Both arms use source 05c71d2 through the same managed Ollama provider with local
Qwen3.5:9b, thinking disabled, temperature zero, an 8,192-token budget, and a
180-second timeout. Exact model/configuration hashes are in preparation.json.
No facts were committed. This is a controlled prompt comparison, not a
full-module ablation, a service lifecycle run, or a native-host quality test.

The cases were authored without inspecting these prompts or model outputs.
The pre-execution review identified P07 and N10 as unsuitable for binary
abstention scoring; it distinguished retained uncertainty in P05 from transient
speculation in N02. Those qualifications and the original cases remain intact.
No aggregate semantic accuracy score was preregistered. Runtime validity is
not semantic correctness, and semantic review remains a separate artifact.

This reruns every case from the completed thinking-enabled comparison, changing
only think from true to false. It is diagnostic reuse, not unseen holdout data.
Manual inspection finds both arms dropping P02's release gates, P05's explicitly
retained uncertainty, and P09's unconfirmed migration state. The candidate in
P01 and both arms in P03 also propose the unrelated existing control fact with
source-event IDs that do not support that fact. Structural validation accepts
those proposals; it does not establish evidential support. All final proposals
remain visible. The setting is not adopted despite zero runtime errors.

The production prompt has not been changed. These reexecuted authored cases
does not establish real-world accuracy or close T18.
