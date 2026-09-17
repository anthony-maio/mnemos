# Hermes configured-registry discovery repair

The old provider rejected registration before initialization even when
`MNEMOS_SCOPE_REGISTRY` named a valid file. Its availability check required an
explicit constructor path, while the registration callback constructs the
provider without arguments. Commit `2360891` resolves the shared default path
at construction and validates the configured file before reporting availability.
Explicit paths still take precedence; injected registries need no file.

The same isolated WSL probe exercised two clean frozen sources through
Hermes' actual plugin loader. Source `554a251` listed the disposable plugin
but rejected it with `Mnemos scope registry is not configured`. Source
`2360891` loaded an available provider with the same valid registry setup.
Both reports passed their expected assertions. The old case reproduces the
defect; its passing probe status is not a product acceptance result.

The probe verifies exact adapter imports and raw/normalized source hashes,
temporary plugin discovery, absence of inherited bridge configuration, and
the expected availability reason. Socket connections and service client
methods were blocked; the service-call counter remained empty. Hermes reported version `0.21.3`
at source revision `36842e639a178741b70c821e0f4cdd129cd0dfe3` during these checks.
This is a fresh observation; earlier host records used a different revision.

The fixture uses a disposable plugin wrapper around the real registration
callback. It does not validate wheel entry-point discovery, service readiness,
native model inference, conversation capture, or cross-host recall. Full
verification of the repaired source is recorded separately when complete.

The first v2 probe failed because of helper payload interpolation. Its exact
payload, logs, and failed report are preserved under `harness-failure`; the
original runner bytes were not retained before correction. The corrected
runner and both passing reports are preserved under `pinned`. The initial
working-checkout negative and positive reports are also retained; their
original negative helper bytes were not preserved. Those preliminary reports
do not replace the later frozen-source comparison.

`evidence-index.json` records the SHA256 of every retained file. No live Hermes
configuration, installed environment, or memory data changed.
