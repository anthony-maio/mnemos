# Inspectability And Consolidation Demo

This walkthrough shows the two product-shell surfaces that should build trust fastest in `0.7.0`:

- inspectability
- consolidation visibility

## Inspectability

Mnemos should not feel like a black box. The home screen and CLI should let you answer:

- why was this memory stored?
- why would this memory be retrieved?
- what changed during consolidation?

### CLI example

```bash
mnemos inspect <chunk-id> --query "python tooling" --current-scope project --scope-id repo-alpha
```

That path should expose:

- scope and scope id
- provenance
- revision history
- graph neighbors
- retrieval explanation for the supplied query

### UI example

From `mnemos ui`:

1. open the Memory panel
2. select a chunk
3. enter a query in **Explain Recall**
4. review the retrieval explanation and consolidation metadata

The UI should show:

- why the chunk was stored
- why it matches the current query
- whether it came from sleep consolidation
- whether recall gating contributed support metadata

## Consolidation

Consolidation should be visible as a product behavior, not an invisible background claim.

Use the UI’s **Run Consolidation** action when you want to inspect the current episodic-to-semantic transfer.

What to look for:

- last consolidation result
- facts extracted
- chunks pruned
- recall-supported vs recall-filtered counts in the current UI session

When **Stable Memory** is active, recall-gated plasticity should make it obvious that not every extracted fact gets promoted into long-term memory.
