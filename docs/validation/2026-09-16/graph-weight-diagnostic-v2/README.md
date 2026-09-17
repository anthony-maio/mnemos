# Graph ranking sensitivity on development cases

The strict v2 run used five frozen development cases and four fixed graph
weights: 0.0, 0.1, 0.2, and 0.4. It copied the previously recorded capacity-16
stores, kept candidate membership and activation fixed, and scored every
candidate with its own graph energy. Every production top-five list matched
the recorded score ordering. No model or network calls were made.

In the one structurally eligible case, target rank was 34, 33, 16, and 3.
The other four cases had no usable target path and remained ineligible.
This separates ranking sensitivity from missing graph connectivity. It does
not establish a general benefit or justify changing the default graph weight.
Production settings, held-out datasets, and acceptance gates are unchanged.

V2 rejects existing output, checks actual d94 Git HEAD and clean status before
and after, and records matching hashes for 107 tracked package/benchmark files,
the frozen vectors, and the capacity report. Its executed helper is preserved.

Exploratory v1 is retained separately. It permitted reusing output directories
and lacked before/after Git and full source checks. Earlier attempt history
could not be established from its retained files. Its report is not accepted
as independently verified provenance; v2 supplies the bound result. Neither
run closes held-out graph coverage or utility acceptance.
