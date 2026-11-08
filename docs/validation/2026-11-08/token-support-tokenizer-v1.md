# Exact token-support prompt preflight

All 52 prepared source-witness prompts passed exact tokenization, ranging from 545 to 735 tokens. The explicit raw ChatML frame includes the disabled-thinking suffix. Each count allows one output token within the fixed 8,192-token context; no prompt or comparison cohort was truncated. Token sequences and rendered prompt hashes are retained.

Eight model-free framing tests passed, including preservation of source text and rejection of reserved model delimiters. The future inference request must use the exact counted text with raw mode enabled. This is a prospective request format; the earlier capability probe used automatic templating and does not establish semantic equivalence.

The first direct-runner startup timed out before processing any prompt. Its owned process was stopped, port closed, and original logs retained. A separate attempt used the previously working Ollama-managed model-loading path with the same model, prompts, tokenizer options, and context limit. That attempt passed. The underlying standalone-startup cause is unresolved; the successful second attempt does not erase the first failure.

The preflight bound model bytes, both executables, runtime DLLs under the installed inference directory, renderer, helper, prepared inputs, and dependency versions. It verified the actual child command and model path before tokenization. Labels were hashed for integrity but never parsed. Empty model-load and unload requests generated no answer tokens; no completion request was issued.

Afterward, the model was unloaded, both owned processes were absent, both ports were closed, and all bound runtime and prepared files were unchanged. The driver exited zero. This establishes input-size feasibility and tokenizer identity for the prepared requests, not semantic quality or retrieval latency. The semantic inference runner and final execution freeze remain pending.
