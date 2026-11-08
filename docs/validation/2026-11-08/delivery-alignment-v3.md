# Delivery evidence alignment v3

Prepared documentation now distinguishes the frozen canonical wheel from the later isolated queued-capture package. The latter passed 107 installed checks, but its measured Python/native command p95 values of 598.64/621.55 ms still fail the original 100 ms requirement. No queued runtime or identity cache code was integrated into canonical delivery, and the optional pilot installation is unchanged.

The README and evaluation page also record the rejected identity-coverage study: 42/72 correct operations, 73/198 correct relations and two unsafe target choices. They retain the stronger historical pairwise result and its six unsafe choices. The offline label-mapping bound explains why changing only the combination rule cannot repair the recorded failures. These remain development results, not fresh acceptance.

Eleven evidence files were copied byte-for-byte into canonical documentation and three receipts into the website. All twelve existing documentation checks passed with the website configured. Ten evaluation receipt links resolved locally, and seven new displayed metrics matched their source receipts. New report-relative links and diff checks passed. The first standalone verifier attempt failed on default Windows decoding of the existing README; explicit UTF-8 resolved that read without changing the source.

Canonical documentation commit: `72be97c0a3fc8e11ab652f7a69506b5f1b4a6726`. Website commit: `f4a2e26105d2cf8628588e7050afb4a69787a58c`. Both worktrees were clean after commits. No runtime, version, install command, live configuration or pilot database changed. The original studies and private archives remain unchanged.

This is local preparation. No rendered-browser check, deployment, publication or upstream acceptance is claimed. The full six-priority goal remains open, including passing quality and complete latency requirements, fresh acceptance and actual daily-use evidence. See the [verification receipt](delivery-alignment-v3.json).
