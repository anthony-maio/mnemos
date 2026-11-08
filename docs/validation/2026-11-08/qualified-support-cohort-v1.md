# Qualified support development cohort preparation

Prepared 12 authored synthetic scenarios with 25 fact/source pairs and 24 questions. The separate source labels contain 14 supported and 11 unsupported pairs. Answer labels contain 12 answerable and 12 absent questions. Runtime snapshots use opaque IDs and preserve complete facts, retained sources, project scopes, and the explicit reference date.

The model-free preparer checked label membership and completeness, then exercised the actual selector using synthetic token responses. It enumerated every supported/unsupported/unresolved annotation combination without passing gold labels to the selector. This produced 633 unique raw prompt frames: 25 source qualification, 50 baseline, and 558 annotated answer frames. Every frame passed delimiter and request-byte preflight. No model inference or tokenization ran.

Prepared inputs, separate grading, the complete prompt universe, baseline request mapping, fixed execution schedule, and content hashes are saved in `.venv/qualified-support-cohort-v1`. The preparer is `.venv/prepare-qualified-cohort-v1.py`; the authored source fixtures are tracked under `benchmarks/local_memory/fixtures/qualified-support-development-v1*.json`.

The planned comparison has 200 semantic calls without failures across baseline, cold qualification/answer, and warm answer arms. The 24 warm repeats do not add independent semantic cases. The fixed arm order can affect timing through provider caching, so it cannot isolate cache speedup. Exact model tokenization and the runtime freeze remain pending. These snapshots do not test extraction, retrieval, live revision races, or native hosts, and they are not fresh independent acceptance evidence. Earlier failed cohorts and their scores are unchanged.
