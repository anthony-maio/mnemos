# Functional plugin repair in local canonical code

The local canonical code checkout now contains source repair `d854db7e`,
fast-forwarded from `3c833caf`. Both MCP and hook command overrides use the
selected interpreter from PATH. The [runtime guide](../../plugin-runtime-python.md)
also explains missing-runtime diagnostics and nonblocking hook skips.

The integration uses the exact revision covered by the
[repair results](functional-plugin-python-v1.md): 38 wrapper checks in each of
six environments and five installed-package checks on Windows Python 3.12.11.
The installed checks contain four MCP roundtrips and one config/guide check.
Those completed tests were not repeated for a fast-forward of identical code.
The previous full-suite result remains bound to its earlier revision.

The checkout was clean before and after the source fast-forward. Eight known
host/pilot configuration files, 148 matching Windows process commands and six
WSL process commands showed no active consumer of this checkout. That scan
covers those files and processes; it cannot enumerate every possible consumer.
All fourteen watched configuration and inactive-pilot paths remain unchanged.

The three original repair protocol/report/receipt files are copied unchanged.
The README now explains that the MCP server owns runtime installation and
hooks skip until it is ready. These are documentation changes following the
verified source fast-forward. The broader legacy experiments remain outside
canonical delivery.

The [receipt](functional-plugin-integration-v1-receipt.json) records the source
and wheel bindings. The 96-file memory package, wheel, manifest, hook
configuration and website are unchanged. Nothing was pushed, published,
deployed or enabled in a host.

Full native hook validation, quality and timing requirements, and the actual
two-project pilot remain open. Both pilot scopes still have zero observed
days. This integration does not change any failed gate or authorize activation.
