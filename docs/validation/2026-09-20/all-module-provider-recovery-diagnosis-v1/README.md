# All-module recovery failure diagnosis

The preserved run used Qwen3.5:9b, think=True, temperature 0 and 8192 output tokens. Of six observed responses, three unstructured predictions returned final text. Two structured chat calls returned empty final content at the 8192-token limit with done_reason=length; one returned a final no-durable-facts response. Nine embedding responses were recorded.

The installed provider rejects empty final content with MnemosExternalServiceError. This explains the failed user extraction without attributing it to embeddings, SQLite, or a valid decision to discard the user fact. The archived database was not changed.

Next: a disposable, source-bound replay with thinking disabled, preserving the changed chat/generate route as a confound. No successful recovery, lifecycle result or production configuration change is claimed.
