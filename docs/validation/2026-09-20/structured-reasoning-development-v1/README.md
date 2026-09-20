# Structured reasoning provider and managed generation settings

The [API diagnostic](../ollama-reasoning-channel-diagnostic-v1/README.md)
reproduced empty final responses when Qwen3.5:9b used constrained generation
with reasoning enabled. Chat with a sufficient token budget returned a valid
qualified historical fact. The production provider change at `e28a846`
selects chat when structured output and an explicit reasoning mode are both
requested. It reads only final message content, preserves request model and
generation options, and reports missing or invalid final text as an error.

The route regression run first failed 13 cases and passed three. After the
change, all 33 focused route and thinking tests passed. The frozen source
checkpoint then passed 1,992 tests with four skips, formatting, strict type
checking, JavaScript parsing, wheel/sdist build, and package metadata checks.
This archive contains all six source/package records. An isolated installed
check was not run for this intermediate checkpoint.

The managed runtime still had no configuration path for the larger budget:
it constructed OllamaProvider with its fixed 512-token, 30-second defaults.
Commit `05c71d2` adds validated max_tokens, timeout, and temperature fields
under providers.ollama, wires them through build_llm_from_settings, and
preserves the old defaults. The initial settings run failed 12 cases and
passed one. All 75 focused settings/runtime/provider checks subsequently
passed, including save/reload and invalid-value rejection. Formatting and
test-import cleanup precede the frozen source copies included here.

Full source/package and installed verification of `05c71d2` remain separate
pending work. No live settings, default inference budget, production extraction
prompt, or installed host configuration changed. Implicit model reasoning
defaults retain the previous generate route; the new chat selection requires
an explicit true or supported reasoning level. Real-model content quality,
all-module benefit, latency, and operational acceptance remain open.

The evidence index binds 28 files: raw regression outcomes, the intermediate
six-check checkpoint, exact committed source copies, and collection helpers.
This README is outside the index. The failed and incomplete model experiments
remain preserved in their original archives.
