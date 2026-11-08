# Explicit Python selection in the plugin wrapper

When `MNEMOS_PLUGIN_PYTHON` is a command such as `python`, the wrapper resolves
it against the supplied `PATH` and launches the resulting absolute path. On
Windows, launching the bare command can select the base interpreter instead
of the virtual environment at the start of `PATH`, leaving the MCP server
without its installed dependencies.

The wrapper searches only listed directories, preserves virtual-environment
symlinks and reports a missing command before launching the server. An explicit
absolute or relative path is retained. Leaving the override unset still uses
the plugin's managed environment.

The selected interpreter must already have Mnemos and its MCP dependencies
installed. This override selects an interpreter; it does not install those
dependencies or activate a host integration.
