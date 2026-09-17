# Claude v16 post-maintenance retrieval diagnosis

The exact comparison uses the preserved v16 direct query, reconstructed from
the corrected target's hash-verified content. It runs the installed `56182d9`
engine against two separate copies of the closed fixture database. Interpreter,
import location, 56 wheel members, and 53 pinned source files were verified.
The source database hash was unchanged and no nonempty source WAL was present.

The target remains stored at version 2. The normal top-5, 1200-token request
returns two other rows. The broad diagnostic top-100, 32000-token request returns
all seven rows, with the target seventh. All seven embeddings are the fixture's
same constant unit vector, producing equal scores and an ID-order tiebreak.
The first selected row consumes 1087 budget units, the second 103; the target
requires 129 and cannot fit in the remaining ten.

This explains the fixture's ranking and context-budget omission. It is not
evidence of durable fact loss, a passed native restart/forget sequence, or
real embedding quality. The native v16 attempt remains failed. A subsequent
transport fixture needs query-discriminating embeddings while retaining its
retrieval limits and lifecycle assertions. Production ranking is not changed
to accommodate this artificial tie.

The exploratory comparison is preserved separately. It used the native MCP
query rather than the later direct query and an engine-only 100000-unit budget.
Its result is not treated as an exact reproduction or as an API request.
Neither comparison made a model call. The original native turn cost recorded
in the reports is 0.0535572 USD; this is the same turn, not another charge.
