# Startup exit diagnostic v2

This archive preserves one bounded diagnostic against the installed `mnemos-memory` 0.7.0 package from source commit `a607a5a6b0a8c2d4f0c4ed00e3b21167cf96f854`. The wheel SHA256 is `4ffcc30358ed95d885f8e86368510bb9de70ec8d3933f75eb12097a3315738ce`; the imported package came from the disposable installed environment recorded in `report.json`.

The probe supplied the same deliberately malformed TOML setup and two-second client startup deadline used by v1. The autostart child exited with code 1 at 655.468 ms. The client returned at 686.713 ms with `status=degraded`, `committed=false`, and `Mnemos service exited before startup (exit code 1)`. The child exit was observed before the response, no descriptor was written, the service log existed, and the child was terminal. Fixture cleanup was verified.

This records numeric child-exit and client-response timing after the installed package change. It does not measure a recall speedup or identify the cause of the earlier intermittent startup failure. The earlier bounded probe is preserved at `../startup-exit-diagnostic-v1`. No model, native host, network, authentication, live configuration, or live database was used. The report and runner are copied byte-for-byte from the disposable run.
