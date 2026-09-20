# Native Hermes review-rejection package discovery preflight

The isolated pinned Hermes host discovered all eight tools from the verified
`db21750` wheel and shut down without a model request, tool dispatch, bridge
call, or service mutation. Discovery took 2.701 seconds.
Live host configuration and memory were unchanged.

Runner and protocol were committed at `dfb08d5687670c57c4e7f38ad9ee69f7f3803d8e` before execution.
Installed package bytes, source and wheel bindings, host revision, compatibility
patch, and live dependency inventory passed the preflight guards.

The included offline report passes 182 existing queue, trace, cleanup,
request-observation, readiness, and cohort cases. Black and strict Python 3.10
mypy pass the two new runner files. Comparison with their predecessors shows
only experiment labels, package bindings, and versioned entrypoint changes;
prompts, evaluator, deadlines, and tool schemas are unchanged. These existing
tests target earlier runner versions; the new runner is also checked by its
binding-only diff and this installed import preflight.

This is host discovery evidence. The separately scheduled native v13 cohort
must be interpreted from its terminal report; discovery is not a lifecycle pass.
The evidence index binds raw files. This README is outside that index.
