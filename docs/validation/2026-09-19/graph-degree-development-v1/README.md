# Graph degree diagnostic: report-parser failure

This first attempt stopped after measuring the first degree-four development
case. The outer report parser expected the flattened evaluation field
`actual_selected_seed_id`, while the production measurement helper returns it
under `query`. The raw measurement and traceback are preserved with the
executed helper and protocol. No model call was made.

The [v2 successor](../graph-degree-development-v2/README.md) corrects that
field access and two related nested production fields. It runs every planned
case and degree in fresh storage under the same protocol. This failure is a
harness error, not a negative product observation or a completed comparison.
