# Current installed recall diagnostic

One diagnostic call completed on installed f1314f6d after three warmups, using
real offline BGE embeddings, exact scoped SQLite retrieval and a disposable copy
of the existing 10,000-accepted-fact synthetic fixture. No support judge, Jev,
automatic modules, live memories or host configuration was involved.

| Interval | Milliseconds |
| --- | ---: |
| Complete client.retrieve | 99.8622 |
| Service submit | 90.2629 |
| LocalMemoryService.retrieve | 89.5497 |
| Engine.retrieve | 88.2422 |
| BGE vectors | 5.0466 |
| Semantic store retrieval | 33.0498 |
| Lexical store retrieval | 35.1165 |
| Batched access update | 2.5748 |

Intervals nest and must not be added. Five facts returned without degradation;
each evidence read took 0.013-0.098 ms. Timing wrappers called the unchanged
implementations. The installed 99-file manifest and source fixture's database,
registry and request log hashes matched before/after. The owned service stopped
and removed its descriptor. No process remains.

This single sample has no percentile or latency-gate claim, no support-quality
claim, no controlled old/new comparison and no causal explanation of the older
slow measurements. It justifies measuring the complete proposed resident path
instead of treating an arbitrary component allocation as the user requirement.

Private evidence: `C:/Users/Ant/AppData/Local/Temp/mnemos-improvement-checks-20261106/profile-current-recall-v1/profile.json`
and `result.json`. Helper: `.venv/profile-current-recall-v1.py`. These are
diagnostics, not a newly frozen acceptance artifact.
