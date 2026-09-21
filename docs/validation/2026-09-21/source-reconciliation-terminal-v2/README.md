# Reconciliation development v2

Candidate 3dac242 restricts support to unique normalized matches, blocks duplicate user-fact additions, and clarifies operation instructions. Same frozen cases, model digest, provider source, settings and expectations as v1. Nine provider cases returned valid staged decisions; foreign scope was rejected before a provider call. Six of ten cases match the original expectations, including deliberately unresolved outcomes and a known contract gap. No production writes or adoption.

Exact support is repaired in the single and multi-candidate cases. New independent facts still become unresolved, including when no target exists. The multi-candidate case remains incomplete. Duplicate correction targets still become conflict rather than unresolved. Assistant disagreement now remains unresolved instead of a conflict, an allowed but incomplete outcome. Known-target correction continues to select the expected target and version. Standalone rejection remains unrepresentable as an add.

The combined experimental suite passes 96 tests and the reconciliation module passes type checking in the Python 3.10 compatibility environment. The initial type-check invocation in the primary environment encountered an incompatible installed NumPy stub; the compatibility environment is the established check used here.

Next experiment moves the same operation contract outside the data JSON. It must preserve the original cases and expectations. These reused development results cannot close fresh acceptance or production integration.
