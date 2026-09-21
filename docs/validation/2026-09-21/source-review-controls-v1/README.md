# Scoped source-review controls

Service and MCP tools now list, inspect and discard extraction reviews using the
host-resolved profile and project/workspace/global scope. Caller-supplied scope
identities are rejected. Inspection resolves the current exact source excerpt
only while source, target and suppression checks pass. Closed or invalid reviews
return no excerpt. Listing and inspection do not write to SQLite, including when
the underlying source changed after extraction.

Discard is durable and idempotent. It releases an open review's source-retention
hold without adding or changing a fact. Pruning and forget can persist validity
changes; read operations only report effective validity.

Validation: 36 focused tests passed, including review storage/restore, service
integration, MCP annotations and compatibility. Python 3.10 type checking passed
for the six changed implementation/control-test files. The stale-source test
compares the full SQLite dump and total_changes before and after both reads.
MCP coverage uses real tool handlers with an in-process service client; it does
not establish native-host or network transport acceptance.

Review resolution remains open. These tools do not promote ambiguous text into
accepted facts. Full-suite, installed-wheel and real-model acceptance must still
be repeated for the finished implementation. No live settings or data changed.
