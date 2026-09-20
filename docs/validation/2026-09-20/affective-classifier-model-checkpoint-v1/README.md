# Affective classifier model selection package checkpoint

Frozen product `ac1a83854dc6734511365aeda6fb92e066e1ce0f` passes 1,962 source tests
with 4 skips, all six source/package checks, 761
isolated installed cases with 4 matching skips, and seven replay
assertions. Individual case outcomes, module origins, and installed package
bytes match the frozen source and wheel.

Wheel: 307,310 bytes; SHA-256 `2b231d82908f9dc990db6f617c057136383cc9b8c5c539267ff631e54e7a8e6f`.

The Python classifier now honors an explicit model selection on Ollama and
OpenAI-compatible providers. Default selection inherits the shared model;
classification does not mutate that shared default. Unsupported overrides
and blank names fail visibly. An unavailable selected model produces no
measured state and does not silently fall back to another model.

The [development record](../affective-model-selection-development-v1/README.md)
preserves 18 initial failures and the focused passing checks. These use HTTP
transport fixtures, not actual classifier inference. The
[configuration note](../../../configuration/affective-classifier-model.md)
explains the saved-value change and custom-provider contract. Managed runtime
settings still use the shared LLM and expose no separate classifier setting.

This package fixes model selection, not classifier accuracy. Statistics
identify configured selection rather than observed model revision. The
archived 33-of-300 fidelity result and all incomplete matrix observations
remain unchanged. Hermes host compatibility, broader recovery, real-model
quality, Desktop, and the operational pilot remain open.

The evidence index binds copied raw records and helpers. This README is
outside that index.
