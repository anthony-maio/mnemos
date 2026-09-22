# Handoff: typed judgment layer and the first judged held-out run

Status as of 2026-09-22, end of day. Branch `feat/typed-judgment-layer` on GitHub, cut from the local worktree branch `wt/functional-local-memory` at `D:\Development\mnemos\mnemos\.worktrees\functional-local-memory`. Nothing here is merged to `master`, released, or pushed anywhere else. Where a number is quoted, the folder it came from is linked; treat anything unlinked as my reading, not a measurement.

## What this branch is

`master` is the pre-September codebase. This branch carries the functional-local-memory work Codex built from [the 2026-09-16 plan](../plans/2026-09-16-functional-local-memory.md) (shared loopback service, durable jobs, scoped retrieval, source-bound extraction, host adapters, held-out harness), the 2026-09-21 review fixes, and the 2026-09-22 judgment layer described below. About 600 commits ahead of `master`. The earlier [research handoff](substack-local-memory.md) covers the state through 2026-09-21 and is still accurate for everything it touches.

## Landed 2026-09-22 (fourteen commits, `32192e7` to `ad5da63`)

An optional typed judgment layer. A `Judge` answers `Noul` (yes/no probability), `Choice` (one option with a distribution), and `Score` (ordered levels) questions; it never generates text. Two backends: `TypeSafeJudge` (hosted Jev, `mnemos/utils/typesafe.py`, httpx, write firewall required at construction) and `OllamaJudge` (local structured output over the same questions, `mnemos/utils/ollama_judge.py`). Everything generative stays on the local LLM and nothing was removed. Defaults never touch the network: `judgment.provider` is `none` unless set.

Where the judge is used when configured:

- `extraction_mode=judged`: segmentation Nouls, then per range a `kind` Choice, a `retention_request` Choice, and a `revises` Noul (`mnemos/extraction/judged_classification.py`); reconciliation as `relation` and `referent` Choices (`judged_reconciliation.py`). Retained text is always a verbatim slice of the source.
- `surprisal.mode=novelty`: one `novel` Noul replaces next-turn prediction.
- Affect: three five-level Scores; with TypeSafe the recall path reads the last encoded state instead of making a round trip (`query_state_source=recent_state`).
- Reconsolidation: `outdated` Noul plus `replacement` Choice over user-role units, correction as an exact span.
- Harness: `--judge typesafe|ollama`, `token_cost` populated, hosted model identity labeled `vendor_version_string` in provenance.

The one design finding worth carrying forward: quote the judged text inside each question's instructions. Ranges that only sit side by side in shared state blur into coin flips (`0ad22bc`).

Also landed: `mnemos_capture` and the Hermes memory-write mirror now default the event role to `assistant`; `mnemos_store` stays `user` (`ad5da63`). A Hermes write already mirrored under `user` will be mirrored once more as a new event after upgrade, because role is part of event identity.

Verification at `ad5da63`: full suite 2554 passed, 4 skipped; black clean; mypy clean on the py3.10 compat venv.

## The spike and the held-out run

[Spike v1/v2](../validation/2026-09-22/typesafe-spike-v1/README.md) on the 46 v12 classification cases: Jev 0.891 / 0.870 presence agreement against Gemma 27B's 0.860, 390 ms median, $0.004 for the set, zero contract errors. That passed the go/no-go and Phases 2-4 followed.

[Held-out matrix, judged on TypeSafe](../validation/2026-09-22/heldout-judged-typesafe-v7/README.md), 4,200 observations in 91 minutes on one A100 for $1.56. Read that README before deciding anything about judged mode. The short version:

- In this harness the `sleep_consolidation` switch is what turns extraction on. Baseline stores every event verbatim. So the paired comparison is judged extraction against verbatim storage.
- Judged extraction doubles `supported_precision` (0.285 to 0.552) at identical retention on the sessions where it completes.
- It fails every correction session (0/60 committed) because Jev encodes the capture `For <context>, keep this in mind: <fact>.` as `no_durable_facts` 85% of the time. Those 60 observations are excluded from the paired comparison, which is why the retention tie flatters it.
- Abstention drops 4.2 points and I have not explained that yet.
- The other four modules made no measurable difference on this corpus. The toggles were verified live from the judge-call purposes.

## Environment and secrets

- TypeSafe key lives in `D:\Development\mnemos\.env` as `TYPESAFE_API_KEY` (that outer repo's `.gitignore` now lists `.env`, uncommitted there). mnemos does not load dotenv; export it as `MNEMOS_TYPESAFE_API_KEY`. In Colab it is the secret `TYPESAFE_API_KEY`.
- Run tests with `MNEMOS_AUTO_START=0` and `--basetemp` pointed outside the repo. Some service tests can reach the live `%APPDATA%\Roaming\Mnemos` if the env is not pinned. Do not touch that directory or the live host configs without asking.
- Files on this branch mix CRLF and LF, sometimes inside one file (`pyproject.toml`, `settings.py`, `engine.py`, `CHANGELOG.md`, `README.md`). Scripted edits must detect the newline per passage; test fixtures that hash bytes must use `write_bytes`.
- The worktree holds untracked validation leftovers from earlier Codex sessions under `docs/validation/2026-09-16` through `2026-09-20` (logs, sqlite WAL files, raw outputs). They are not on this branch on purpose.

Commands, from the worktree:

```
MNEMOS_AUTO_START=0 .venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp <scratch>
.venv/Scripts/python -m black --check mnemos tests benchmarks
.venv/compat-py310/Scripts/python.exe -m mypy mnemos
.venv/Scripts/python -m benchmarks.local_memory.typesafe_spike --judge both
.venv/Scripts/python -m benchmarks.local_memory.prepare_colab_bundle --extraction-mode judged --judge typesafe --surprisal-mode novelty
```

Colab: upload a bundle folder as-is (keep `source.zip` zipped) to `MyDrive/mnemos-eval/`, open `benchmarks/local_memory/colab/heldout_colab.ipynb`, set `BUNDLE_NAME`. Profiles run concurrently and resume from Drive checkpoints. Two bundles are prepared at `D:\WindowsTemp\mnemos-colab\`: the judged one above (done) and `mnemos-heldout-3b3d19d-v7-source-bound` (local Qwen3.5 9B, not yet run; expect 15-25 hours on an A100).

## Next, in order

1. Fix the `keep this in mind` misclassification. Add those cases to the spike inputs, revise the `kind` option descriptions in `judged_classification.py`, confirm on the spike, rerun the `full` profile only (90 minutes, about thirty cents).
2. Explain the abstention drop: diff what `full` retrieves on `no_answer` queries against baseline in the raw outputs on the Drive mirror.
3. Add `supported_precision` and the correction commit rate to the notebook's printed summary and to `summary.json`.
4. Run the source-bound bundle so there is a like-for-like local comparison. Extraction is the whole result on this corpus, so `baseline` vs `full` across the two runs is the comparison that matters.
5. Phase 6 from the plan, still not started: merge novelty into the classification call, `judge_rerank_enabled`, honor `Retry-After`.

## Decisions waiting on Anthony

- Corpus v7 vs authoring v8. The v7 prefixes (`keep this in mind:`, `A related decision detail:`) penalize both extraction paths on exact-text retention and trip the judged `kind` question.
- Whether judged mode becomes the recommended default. Not on this evidence: precision is real, the correction failure is disqualifying until fixed.
- Website edits in the outer `D:\Development\mnemos` repo (`web/index.html`, `web/style.css`) are uncommitted and waiting for review.
- The partial ~10 GB backups need an integrity check and an OK before anything is deleted.
- T17 native hosts, the pilot, and the T19 claims remain open from the earlier handoff.

## Commit hashes on this branch versus the local worktree

Pushing required moving three `docs/validation/**/raw-evidence.zip` files (one is 145.7 MB, over GitHub's limit) into Git LFS, which rewrote every commit since `origin/master` on this branch. The local `wt/functional-local-memory` branch keeps the original hashes and the trees are otherwise identical. [2026-09-22-lfs-rewrite-hash-map.txt](2026-09-22-lfs-rewrite-hash-map.txt) maps all 609 old hashes to new ones. The ones cited above: `32192e7 -> 6ffd45e`, `0ad22bc -> 0f30512`, `3b3d19d -> 5eda261` (the Colab bundles and manifests bind `source_commit` to the old hash), `e884507 -> 2510443`, `ad5da63 -> 6dc614b`, `e6887de -> 082661e`, `ac66088 -> 36a9838`. Validation READMEs written before this push cite the old hashes throughout; resolve them locally or through the map.

## Rules that applied to this work

Commit on the branch, never push or merge without asking (this push was asked for). No AI attribution in commits. Cloud judgment is an explicit opt-in, never a silent fallback when local fails. Retained memory text is always a verbatim slice, so the worst judged-mode failure is a wrong label or a dropped fact, never invented content.
