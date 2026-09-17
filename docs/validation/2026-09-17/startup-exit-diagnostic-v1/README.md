# Startup exit diagnostic

This archive preserves one bounded diagnostic against the verified installed `mnemos-memory` 0.7.0 package. The package was imported from `C:\Users\Ant\AppData\Local\Temp\mnemos-installed-lost-ack-v3\Lib\site-packages`, with wheel SHA256 `f4e270cb2c0469d42a65b1152684297fa9fc61af190dacde8890ab50c3d90b14` and source commit `6cde46561c0c42c0d6124d86a09d1e2832542c9d`. This is before any later startup-response repair.

The probe supplied a deliberately malformed TOML file in a fresh private fixture and used a two-second client startup timeout. The autostart child exited with code 1 at 652.265 ms. The client returned at 2,048.842 ms with `status=degraded`, `committed=false`, and `Mnemos service descriptor is unavailable`. The descriptor was absent, the service log existed, and the child was terminal before the client returned. The private fixture was removed after observation.

This demonstrates the client wait behavior when the child exits before writing a descriptor. It does not establish that this behavior caused the unrelated full-suite startup observability failure, whose cause remains unknown. No model, native host, network, authentication, live configuration, or live database was used. The report and runner are copied byte-for-byte from the disposable run.
