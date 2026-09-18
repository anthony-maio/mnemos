# Supplied outbox selection

An explicitly supplied empty Outbox was discarded because the constructor used
truthiness to select its default. Outbox defines length as its pending event
count. This sent a capture to the default file instead of the caller's selected
recovery file, and replaced a supplied PII-blocking firewall with the default
redacting policy. The first regression run produced two failures and one passing
nonempty-outbox control.

The client now selects a supplied object with an explicit None check. Its path
and policy no longer depend on whether it contains pending events. All 34
focused client, service, lost-acknowledgement, and interrupted-capture tests pass.
The three new cases cover empty/nonempty selection, a new client's replay of the
chosen file without duplicate dispatch, and preservation of stricter PII blocking.
Replay in the new selection tests uses real HTTP with a recording dispatcher;
it is not new SQLite extraction or native-host evidence.

Source snapshots, the patch, JUnit results, and the negative run are preserved.
Full frozen-source and installed-package validation follow separately. The frozen
Ollama evaluation, live host settings, and live memory data were not changed.
