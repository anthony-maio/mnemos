# Affective classifier model selection development

The configured classifier model was ignored. This repair sends an explicit
model on supported Ollama and OpenAI-compatible request paths, preserves
shared-provider defaults, and rejects unsupported overrides. Omitted model
selection inherits the supplied provider. Statistics identify configuration
selection without claiming an observed model revision.

The original 18 failing regression cases are preserved with their exact test
input. The first focused run passes 45 cases; the expanded run passes 83.
HTTP transport fixtures cover concurrent classification/shared requests,
inheritance, unavailable selected models, unsupported custom providers, and
invalid selections rejected before HTTP. No network or actual model is used.
Full source and installed-package checks are separate and pending here.

This fixes a configuration contract. It does not improve the archived
33-of-300 classifier fidelity result or establish independent calibration.
Managed runtime settings still use their shared LLM. See the
[configuration note](../../../configuration/affective-classifier-model.md)
for the saved-configuration behavior change.

The index binds original test results, candidate sources, and collection
inputs. This README is outside the index.

Subsequent [full package verification](../affective-classifier-model-checkpoint-v1/README.md)
passes on `ac1a838`; the original focused evidence above remains unchanged.
