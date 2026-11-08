# Response-completion package verification

The isolated package from `eba8cfc46965e5347359f3a543373518a00262ec` passed archive integrity, source parity and installed lifecycle verification. All 107 tests passed with imports bound to the fresh installation. All 98 packaged source/resource files matched committed source and the installed copies, including after tests. The sdist had 105 source members. Declared smoke/example resources were present.

Wheel SHA256: `cfe4e9b480fb8232453d53e928a6a2ba840d26bb3a7a9106dc6bdc1968fd4c2e`. Sdist SHA256: `5002d3f8dafcb1117424b399eef8ed1d144cbd76d170996c566af74262b40148`. Build backend: hatchling 1.33.0. The isolated Python 3.12.11 environment used the same dependency versions as package v2. No earlier package was overwritten.

Tests cover response completion, failed writes, lost notification recovery, outbox selection and durable queue receipts, worker/pump lifecycle, real SQLite capture/correct/forget behavior, service errors and related-forget operations. Test files were copied unchanged and hashed; the runner asserts its mnemos import comes from the isolated environment. Temporary fixtures were outside the source repository.

The identity-cache prototype is included as inactive source but remains disconnected from adapters. Live pilot installation and host settings are unchanged. This is local package validation, not publication, native-host acceptance or a command latency pass. The frozen response-completion command timing protocol is the next gate. The companion JSON identifies the retained private archive.
