# Reranker support preparation

Historical preparation checkpoint. The screen subsequently [completed and failed quality](reranker-support-v1.md). The pending items below describe the earlier preparation state.

The prospective protocol was committed at 83f3f201 before preparing requests. Preparation produced 68 known-question records and 123 fact/source witness requests, at most three per question. Each witness includes the original full candidate cohort, including stored conflicts as nonselectable context. Expected labels remain in a separate grading file. Original observations, questions and labels matched their archived hashes before preparation.

Pinned model metadata confirms a standard qwen3 configuration without custom auto_map. The exact model card, configuration and tokenizer configuration were downloaded and hashed. No weights were downloaded and no inference ran. Token framing and limits, a separate CUDA runtime, model loading, decision tests and execution freeze remain pending. The existing QA environment is CPU-only and was not modified; an RTX 4080 was visible during preparation.

Inputs SHA256: 88035b398e23d12c3751331d884d9422a1222daa0b925977f6b7d35ecaf515fe.

Labels SHA256: f2af04ad22b277dc096d9d084b85b871816e514814739673fcf52ed9f790e916.

Private preparation is retained in `.venv/reranker-support-v1`; its receipt records metadata URLs, hashes, protocol binding and helper hash. This is development preparation, not evidence of support quality, calibrated abstention, timing acceptance or pilot use. Earlier failed candidates remain closed, and live configuration and databases are unchanged.
