# Selecting the affective classifier model

The Python API accepts an optional `AffectiveConfig.classifier_model`.
Omit it, or use `None`, to inherit the supplied LLM provider's model. An
explicit model name is sent for classification requests only; other modules
continue using the shared provider's default model.

```python
from mnemos.config import AffectiveConfig

inherited = AffectiveConfig()
separate = AffectiveConfig(classifier_model="my-local-classifier")
```

Ollama and OpenAI-compatible providers support this request override. The
endpoint and credentials remain those of the supplied provider. A custom
provider must advertise `supports_model_override = True` and honor the
`model` request keyword; otherwise explicit selection raises a configuration
error when constructing the router. Blank names are rejected.

If the selected model is unavailable, classification produces no measured
state. It does not silently retry with the shared model or fabricate a
neutral state. Router statistics report the configured selection, not a
verified model revision or evidence of a successful inference.

Older versions defaulted this field to `"llama3"` but ignored it. A saved
Python configuration that explicitly contains that value now requests that
model. Remove the field or set it to `None` to retain shared-provider
inheritance. No existing configuration or memory database is rewritten.

This setting belongs to the Python `MnemosConfig`/`AffectiveRouter` API.
Managed runtime settings do not yet expose a separate classifier model;
they continue to use the configured shared LLM. Request-level transport
tests do not establish classifier accuracy or a real-model quality gain.
