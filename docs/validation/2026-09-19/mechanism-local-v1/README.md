# Local module mechanism result

The held-out phase completed on frozen `fc72cc2` using local Ollama, Qwen
3.5 9B, and Nomic embeddings. All 10 phase artifact hashes were verified
before archiving and every compressed entry was read back byte-for-byte.
Exact digests, settings, paths, and timings are in `phase-binding.json` and
`manifest.json`. The main lifecycle run was still active when this archive
was created; this archive makes no lifecycle or whole-matrix claim.

Graph: 50 cases and 150 observations completed, but zero cases were eligible
against the unchanged minimum of 40. Structural correctness passed, including
144 cycle observations. Graph benefit remains unmeasured. Overall mechanism
gates failed coverage; execution success is not acceptance success.

Affective: 50 pairs, 100 conditions, and 300 observations completed. Ranking
with synthetic gold states raised target top-1 from 0% to 34%; the clustered
paired difference was 0.34, with 95% interval [0.18, 0.50]. This is a scorer
result with supplied gold states, not an end-to-end classifier benefit.
The real classifier parsed all 300 outputs, but only 33 passed the configured
fidelity check. Twenty-one pairs were non-discriminating. Top-5 paired
difference was 0.05 with interval [0.00, 0.15]. Human acceptance is NOT RUN.

`phase-artifacts.zip` retains every original report and raw observation,
including model outputs, embedding vectors, logs, and manifests. The small
reports are also available directly. Negative results and thresholds are
unchanged. No model, evaluation, website, or live-memory settings were changed
to produce this archive.

The first archive preflight incorrectly expected eleven files and stopped
before writing the archive. Its helper and failure record are retained.
Version 2 verified the exact ten-file set and all recorded hashes.
