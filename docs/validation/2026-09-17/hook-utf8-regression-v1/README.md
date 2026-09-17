# UTF-8 hook input regression

The native v5 diagnostic observed a nonempty Stop message, a valid turn ID,
and no reentry, followed by a sanitized UnicodeEncodeError diagnostic and no
capture dispatch. Windows piped stdin on this environment uses cp1252 with
surrogateescape. Reading UTF-8 curly quotes through that decoder can introduce
surrogates that fail when the adapter builds the UTF-8 event identity.

The regression sends UTF-8 JSON containing curly quotes, Japanese text, and
an emoji through a cp1252/surrogateescape text wrapper into each actual adapter
CLI. Both hosts previously lost prompt and assistant capture. A fifth case
shows parsed Codex Stop errors previously omitted nonblocking JSON when the
event name came from the payload rather than argv.

Both adapters now read the binary input as UTF-8, with a text-stream fallback
for embedding callers. Codex error handling uses the parsed event. All five
new cases and related adapter coverage pass: 75 tests. Raw red and green
logs/XML are preserved. Recording clients isolate the decoding and adapter
boundary; these tests do not prove native post-repair acceptance.
