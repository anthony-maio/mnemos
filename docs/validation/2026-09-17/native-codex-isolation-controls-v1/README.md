# Separate MCP and hook shutdown controls

Both controls used the same installed ebe1b0d wheel, Codex 0.146.0 executable, disabled Mnemos modules, deterministic fixture vectors, a disposable SQLite service, and an explicitly unelevated read-only sandbox. MCP-only completed exactly one native health call and exited zero. Hooks-only completed a native model response without MCP tools and exited zero, despite a three-second Stop-hook timeout in the terminal. Authentication copies and service descriptors were absent before archiving.

Selected native records are checked against the original session lines. Independent SQLite reopen records the captured events and roles without treating their presence as proof of hook deadline acceptance.

Together with the two no-integration controls, these observations show that neither isolated configuration reproduced the full continuity run's crash. The hook timeout can occur without a crash. They do not prove that the combination is the cause: full continuity used store/retrieve operations with interactive tool approval, and did not explicitly pin Windows sandbox mode. A matched combined configuration is the next control. No live host configuration or database was changed, and no release was published.
