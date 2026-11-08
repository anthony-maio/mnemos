# Python selection for Claude plugin servers and hooks

Both wrappers resolve a command in `MNEMOS_PLUGIN_PYTHON`, such as `python`,
against the configured PATH before launching it. This prevents Windows from
substituting the base interpreter for the selected virtual environment.
Executable extensions are respected, and Linux virtual-environment symlinks
are preserved. An explicit absolute or relative path remains literal.

If a command is unavailable, the MCP wrapper exits with a diagnostic that names
the override and PATH. The hook reports `runtime_not_ready`, skips with exit
zero and performs no installation. Skipping is not successful capture. Existing
hook handling still prevents a child exit of two from blocking the host.

Without an explicit override, the MCP wrapper continues to own installation
of the managed runtime. Hooks use it only when its existing source stamp is
current. This source-wrapper repair does not alter the frozen memory wheel,
its configuration or its quality and timing results.
