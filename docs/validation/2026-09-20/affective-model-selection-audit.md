# Affective classifier model selection audit

At product `b28778a`, `AffectiveConfig.classifier_model` still promises a
classifier model name, defaulting to `llama3`, but
`AffectiveRouter.classify_state` calls its injected provider without that
setting. Ollama's request payload uses the provider's constructed `self.model`.
This is an ineffective configuration field, separate from classifier quality.

A local recording-provider check supplied `classifier_model="requested-classifier"`
and returned the parseable fixture response `0, 0, 0`. The observed result was:

```json
{"requested_model": "requested-classifier", "provider_kwargs": [{}], "parsed": true}
```

No network or model was used. The first scratch command used an incorrect
module import and constructor name and failed before running the check. The
corrected check used `mnemos.types.Interaction`, `AffectiveRouter(llm=...,
embedder=SimpleEmbeddingProvider(), config=...)`, and a provider recording
the keyword arguments of `predict`. This is a development reproduction, not
a real-provider selection test.

The repair needs an explicit provider/model-selection contract, effective
model provenance, and readiness/error behavior. Merely passing a `model`
keyword is insufficient while a provider ignores it. Preserve inheritance
from the configured shared provider unless a separate classifier is selected;
do not silently redirect the current local workload to another model or cloud.
The [development repair](affective-model-selection-development-v1/README.md)
now implements that request contract. Eighteen initial failures become
passing regressions, and an expanded 83-case focused run passes. Full source
and installed-package validation remain pending at this development checkpoint.
The [configuration note](../../configuration/affective-classifier-model.md)
explains inheritance, custom-provider capability, and the saved-value change.

The [earlier fidelity review](../2026-09-19/affective-fidelity-review.md)
still stands: all 300 outputs parsed, but only 33 met the frozen distance
criterion against two repeated authored vectors. The parser replay changed
zero states. No source defect has been shown to explain that numeric mismatch,
and independent annotation/calibration is absent. This configuration finding
does not improve, invalidate, or relabel the archived fidelity result.
