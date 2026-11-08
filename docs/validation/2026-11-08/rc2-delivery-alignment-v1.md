# Local rc2 documentation alignment

Canonical delivery commit `bc482489` adds the rc2 package and fresh Windows dependency reports to the README. Website commit `eed4555` adds the matching evaluation paragraph and matrix receipt. Both remain local and unpublished.

Five copied canonical reports/receipts and the website matrix receipt matched the original evidence byte for byte. The evaluation page contains one link to that receipt. Its 270 executions represent 90 distinct installed cases repeated on three Windows Python versions, including nine automated MCP roundtrips.

All 12 canonical documentation consistency checks passed in 0.46 seconds with the prepared website configured. JUnit is retained at `.venv/rc2-delivery-docs-v1.xml`; the content/link audit is `.venv/rc2-delivery-alignment-v1.json`. Both repositories passed `git diff --check` and were clean after their commits.

The earlier rc1 benchmark remains failed. This update establishes no new semantic, latency, platform or pilot acceptance. The rc2 artifact remains distinct from canonical service delivery and the activated optional explicit-memory pilot. No live host, database or published page changed.
