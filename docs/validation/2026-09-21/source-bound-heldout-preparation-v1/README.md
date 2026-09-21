# Frozen source-bound held-out run preparation

The fresh v7 corpus is running unchanged against all seven profiles with three
repetitions, for 4,200 planned session/profile/repetition observations. Product
and harness source are frozen at a6b0bf3; source ZIP and file hashes are preserved
in ../source-bound-development-matrix-v1 and preparation.json. The corpus is
../fresh-corpus-preparation-v7 in the September 20 validation directory, bound
by SHA-256 b129004eacc1364dce6b8ab069a0b2b9cdf5604543aae003b8ea9f1e42a74fec.

Providers are local Ollama Qwen3.5 9B and Nomic embeddings; exact digests,
hardware and generation options are recorded. The launcher revalidates model,
source and corpus hashes before starting. The existing minimums, labels,
scoring and gates are unchanged. The preceding one-session development matrix
completed without runtime errors but preserved all exact-text answerable misses.
No positive quality result is implied by launching this run.

The run uses synthetic authored inputs. Exact-text precision is a proxy, not
semantic factual precision. No module-challenge corpus is supplied; separate
module-utility evaluations remain required, including associative retrieval.
This artifact is preparation, not completion or acceptance. No live host
configuration, live memory migration or publication is involved.
