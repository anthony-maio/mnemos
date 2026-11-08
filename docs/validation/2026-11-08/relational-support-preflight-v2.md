# Relational support token preflight

All 68 frozen development requests passed native token preflight. Exact raw frames contain 1,613-2,027 tokens, leaving the full 2,048-token output reservation inside the 8,192-token context. No frame was truncated and no semantic completion was requested during this preflight.

The owned local runtime used the retained Qwen3 14B model and verified Ollama binary. The receipt records native process ownership, executable/library and model-file bindings, exact token IDs and framing options. Empty load returned no generated output. Cleanup verified model unload, parent/native process absence, closed ports and unchanged runtime and prepared-input bindings. The preflight process exited zero.

The first attempt stopped before tokenization on an obsolete helper assertion expecting a field from the previous experiment's request format. It generated no answers and verified cleanup. Both attempts remain preserved. The second helper removed only that obsolete check; exact prompt reconstruction and frozen input hashes remained required.

Token records: `.venv/relational-support-tokenizer-v2/tokens.jsonl`, SHA256 6c9ce316fbe7a86fca524fc6f14e5722f8feed4735a66b6e52b504dd0cece2d7.

Control receipt: `.venv/relational-support-tokenizer-v2/control.json`, SHA256 214a7a3b3a933538da191e4b631687199ad468cb1dba3f7d00d542bace13cc13.

The one-pass execution harness in c972896f passed 31 combined model-free checks in 0.98 seconds. It preflights all reconstructed requests before the first call, records starts and raw replies, preserves completed malformed outputs without retry, and stops after uncertain completion without attempting the next question. These checks do not establish semantic quality or model latency. The separate execution screen must retain its own results, failures and cleanup evidence.
