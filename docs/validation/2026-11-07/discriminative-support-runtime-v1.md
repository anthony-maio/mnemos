# Discriminative support runtime preparation

Eight files from the pinned model revision downloaded successfully into the
private feasibility directory. File sizes match repository metadata, LFS files
match their advertised SHA-256 hashes, and the three previously inspected
metadata files retain their hashes. Only safetensors weights were downloaded.
No model forward pass has run.

The existing isolated QA environment supplies Python 3.12.11, CPU PyTorch
2.10.0, Transformers 4.57.6, safetensors 0.8.0, tokenizers 0.22.2,
sentencepiece 0.2.1 and NumPy 2.5.4. It was reused without package installations.
Model files, cache and preparation outputs use a separate feasibility directory;
the optional pilot environment was not changed. The next worker will use CPU
float32 with no device fallback, as required by the protocol.

Tokenizer loading uses local files and disables remote code. All 241 exact pairs
fit the 512-token cap without truncation. Maximum input length is 89 tokens;
the total is 11,366 classifier tokens. These are not host-context token counts.
The private `model-preparation.json` retains every per-pair count, all model-file
hashes and the environment's installed dependency versions.

The pure benchmark decision rule is now implemented. It retains the neutral
class when computing softmax, validates finite three-class logits, requires all
directed cross-candidate evidence pairs, rejects duplicate witnesses, and vetoes
both candidates when either direction crosses the fixed contradiction threshold.
A candidate that fails support still participates in contradiction checks.
Witnesses and veto evidence identities remain in the result.

All nine focused checks pass, including a supported fact vetoed by an unselected
contradictory source while an unrelated reviewer fact remains selected. Missing
pair coverage, malformed logits, neutral dominance and numerical stability are
also covered. All 68 prepared cohorts pass a separate synthetic neutral-output
coverage check. These synthetic results establish control flow, not model quality.
Black formatted both new Python files; scoped mypy passes for the decision module
with `--follow-imports=silent`, and `git diff --check` passes.

Preparation and validation processes exited successfully. The measured runner,
its prospective source/model/dependency freeze, model loading and warmup, the
single fixed inference pass, independent grading and latency audit remain pending.
Earlier evaluations, product source, installed pilot artifacts, live host
configuration and databases remain unchanged.
