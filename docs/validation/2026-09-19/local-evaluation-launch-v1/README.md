# First local preflight: cold-load timeout

The first local preflight verified the clean `fc72cc2` source, all 18 indexed checkpoint files, all 58 installed package files, the original coordinator hash, and both original model digests. Its generation request used the benchmark's temperature 0, maximum 512 output tokens, thinking disabled, and no model-context override. The request timed out after 120 seconds before Ollama finished loading Qwen.

Selected runner diagnostics show a 16,384-token context, GPU allocation that fits the RTX 4080, a GPU-discovery watchdog warning, and cancellation when the client disconnected. A separate cold-load probe allowed up to 600 seconds for loading only. Qwen completed loading in 149.623 seconds. This does not change the benchmark's 120-second inference timeout or the 500 ms warm-recall target.

The cold-load helper then incorrectly sent Nomic to `/api/generate`, which returned HTTP 400. That helper's overall result remains failed; the completed Qwen load is a separate observation. The [second preflight](../local-evaluation-launch-v2/README.md) used `/api/embed` for Nomic, retained the original inference timeout, and passed. No failed record or executed helper was rewritten.
