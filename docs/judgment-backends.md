# Local judgments and optional Jev

Mnemos keeps memory on the machine. Typed judgments are off by default
(`judgment.provider = "none"`). Ordinary explicit memory storage and retrieval
do not require Jev. Source inspection remains necessary; the current local
answer-support experiments have not met the quality and latency gates.

For local judgments, configure the local Ollama backend explicitly:

```toml
[judgment]
provider = "ollama"
model = "your-installed-local-model"
```

Selecting a model makes the backend available; it does not establish answer
quality. Local failures never trigger a switch to Jev.

For optional hosted judgments, select TypeSafe explicitly:

```toml
[judgment]
provider = "typesafe"
model = "jev-latest"
```

Supply the key through `MNEMOS_TYPESAFE_API_KEY` in that process's environment.
If a private `.env` uses `TYPESAFE_API_KEY`, the launcher must load it and map
that value to `MNEMOS_TYPESAFE_API_KEY`; Mnemos does not automatically discover
that file or alias. Never copy the key into tracked configuration or output.
Having a key does not select the provider. Jev receives the state and questions
for the selected judgment operations over HTTPS after the memory firewall.
Memory storage stays local, but those judgment inputs leave the machine.

Return `judgment.provider` to `none` to turn typed judgments off, or to `ollama`
to use the configured local backend. The provider switch applies to typed
judgments, including any enabled judged extraction; it does not automatically
enable the experimental answer-support checker or change the embedding/LLM
providers. This guide describes the unreleased delivery candidate.
