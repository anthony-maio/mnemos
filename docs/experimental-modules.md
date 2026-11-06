# Experimental module settings

The November 6 development branch defaults all five module switches to `false`.
Provider selection is separate from module selection. Supplying an Ollama or
hosted provider does not enable experiments. This change is unpublished.

The managed service's explicit remember, inspect, retrieve, correct and forget
operations remain available. Explicit writes use embeddings and storage without
model classification or extraction when the modules are off. Queued work still
needs a completed job receipt before it can be described as completed.

For projects where capture is enabled, source events are still retained.
With `sleep = false`, completed capture jobs retain role-qualified captured text.
That text is marked as captured evidence, not an accepted user assertion. Turning
off enrichment does not disable capture or remove existing evidence.

The older embedded Python `process()` API has no durable capture queue. With
`surprisal = false`, it returns `stored = false` for automatic ingestion. Its
`remember()` method still stores explicit text. An enabled sleep experiment can
buffer interactions for later consolidation; with sleep disabled, it collects
no new episodes and `consolidate()` does not run extraction. Disabling
`mutable_rag` also prevents consolidation from processing proposed revisions.

## Opt in deliberately

Set only the modules needed for an experiment. For example, automatic extraction
can be enabled in the existing settings file with:

```toml
[modules]
sleep = true
```

`sleep` enables extraction, `mutable_rag` allows automatic corrections,
`surprisal` enables automatic encoding gates, `affective` enables state
classification/scoring, and `spreading` enables graph derivation/retrieval.
Relevant provider and retrieval-profile settings still apply. A module flag
alone does not establish a working provider or a quality improvement.

Python callers can pass `MnemosConfig(modules={"sleep": True})`. Environment
overrides retain the `MNEMOS_MODULE_SLEEP=true` form, with equivalent names for
the other flags. Settings and environment overrides use the existing precedence.

## Upgrade behavior

Explicit `true` and `false` values are preserved. Omitted flags adopt the new
defaults, including in existing partial configuration files. A configuration
that previously depended on implicit enabled flags must opt in explicitly to
keep that experimental behavior. Reading settings does not rewrite them.

The setup form preserves current module choices when ordinary fields are saved.
Selecting Starter SQLite explicitly sets all modules off. Selecting the embedded
recall experiment enables sleep extraction and its recall-gating setting; the
other module switches remain off. Custom module combinations are not labeled as
Starter SQLite.

Research fixtures and comparison helpers specify their own flags so changing
application defaults does not silently change their experiment arms. Frozen
scores and archived source snapshots remain unchanged. The earlier installed
wheel and native-host receipt refer to their recorded product checkpoint;
validation of that artifact does not validate this later source change.
