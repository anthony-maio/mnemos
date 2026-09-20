# Terminal authored extraction comparison

All 40 planned calls are recorded: 20 synthetic cases, two prompt arms, one
repetition per case and arm. There are 39 runtime-valid responses and
1 recorded runtime errors. No failed call was selectively rerun. The
runtime summary separates the production and history-only arms; raw requests,
responses, failure records, final proposals, and frozen preparation are included.

Both arms use source 05c71d2 through the same managed Ollama provider with local
Qwen3.5:9b, thinking enabled, temperature zero, an 8,192-token budget, and a
180-second timeout. Exact model/configuration hashes are in preparation.json.
No facts were committed. This is a controlled prompt comparison, not a
full-module ablation, a service lifecycle run, or a native-host quality test.

The cases were authored without inspecting these prompts or model outputs.
The pre-execution review identified P07 and N10 as unsuitable for binary
abstention scoring; it distinguished retained uncertainty in P05 from transient
speculation in N02. Those qualifications and the original cases remain intact.
No aggregate semantic accuracy score was preregistered. Runtime validity is
not semantic correctness, and semantic review remains a separate artifact.

The production prompt has not been changed. This one-repetition authored set
does not establish real-world accuracy or close T18.
