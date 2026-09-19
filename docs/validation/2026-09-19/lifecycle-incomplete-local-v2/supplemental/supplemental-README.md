# Supplemental local comparisons queued

A separate supervisor is waiting on the live main coordinator process 19032. Its initial process-bound observation is copied here; current state is at `C:\Users\Ant\AppData\Local\Temp\mnemos-local-supplemental-v1\state.json`. Waiting is not comparison execution or accepted quality evidence.

The supervisor retains an OS process handle and requires a successful exit, the completed five-phase main queue, and matching hashes for every completed main-phase artifact. It then runs the applied encoding comparator and four-strategy retrieval comparator serially. Each stage rechecks clean source `fc72cc2`, exact dataset/protocol hashes, and both model digests. Pre-existing output/work directories cause a stop. Failures are retained in the supervisor state; there is no automatic restart or overwrite.

Both use the frozen `local-memory-heldout-v3.jsonl` dataset, 200 held-out sessions and three repetitions, local Ollama, Qwen `qwen3.5:9b`, Nomic `nomic-embed-text:latest`, and thinking disabled. The frozen comparator code supplies temperature 0, maximum 512 output tokens, top-k 5, retrieval context budget 1,200, and the 120-second provider timeout. Encoding compares full, surprisal-disabled, and the applied length/noise rule. Retrieval compares vector, lexical, hybrid, and full retrieval on identical accepted snapshots. Their output/work directories and manifests are separate from each other and from the main queue.

The exact invocations are in the archived supervisor. `PYTHONPATH` and `MNEMOS_*` variables are removed only from its child execution environment. Source runs use the worktree Python interpreter with the clean frozen snapshot as the working directory. Comparator manifests remain authoritative for code, input, provider, and artifact bindings.

Before launch, PowerShell syntax parsing passed. A disposable external Python process also verified that the supervisor's `Get-Process` handle and `WaitForExit` pattern retains the actual nonzero exit code. The supervisor is not evidence that either comparison has completed. The original benchmark and live host configuration are unchanged.
