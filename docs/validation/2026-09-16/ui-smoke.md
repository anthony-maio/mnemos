# UI smoke validation

Date: 2026-09-16

This checkpoint exercised the loopback UI against a temporary SQLite database,
temporary scope registry, authenticated `MnemosClient`, and the real
`ControlPlaneService`. The fixture is [ui_smoke.py](../../../benchmarks/scenarios/ui_smoke.py).
It creates one enabled project for profile `ui-smoke`, uses
`UnavailableLLMProvider` and `SimpleEmbeddingProvider`, and deletes its
database and registry when it exits. The provider choice makes this a control
plane and persistence check. It is not evidence of extraction quality or a
native Codex, Claude Code, or Hermes session.

The fixture was started with:

```text
.venv\Scripts\python.exe benchmarks\scenarios\ui_smoke.py
```

The browser used the Playwright CLI against the printed `127.0.0.1` URL. The
initial dashboard showed one project chunk. Pause changed the visible count to
zero, and enable restored the project and its chunk. The integration preview
reported the committed project state after enable.

The seeded content included the literal string
`<img src=x onerror="window.__mnemos_xss = true"> Keep this escaped.`. The
memory card and detail panel displayed the tag as text, and
`window.__mnemos_xss` remained `false`. This is browser evidence for this
fixture only.

Selecting the item and using Correct opened the expected prompt. Submitting
`Corrected via Playwright UI` displayed `Correction queued. Processing:
pending.` and one durable job. Running consolidation completed the job and
showed the corrected content with a revision record. The UI's consolidation
summary rendered `undefined facts` and `undefined promoted / undefined
filtered`, while the raw health payload showed a passing maintenance result;
the summary field mapping needs repair. The inspectability subtitle also kept
the stale text `Latest consolidation: none recorded in the current store.`

Forget then displayed `Memory forgotten.` and zero visible chunks. The detail
panel showed `No stored memories yet.`. A restart was not run in this browser
checkpoint because the fixture owns a temporary in-process service; restart
continuity is covered separately by the service and cross-host validation.

Screenshots:

- [initial dashboard](../../../output/playwright/ui-smoke-initial.png)
- [post-forget dashboard](../../../output/playwright/ui-smoke-final.png)
- [fixed maintenance and wrapped path](../../../output/playwright/ui-smoke-fixed.png)

The initial screenshot also exposed a responsive presentation issue: the long
SQLite path in the current-state card overlaps the neighboring health card at
the 1280 pixel viewport. Browser console output contained a favicon 404 and a
password-field-in-form warning. No live configuration, database, host session,
or paid model was used.

After the UI fix, a fresh 1280 pixel reload showed the path wrapped inside its
card. Running maintenance then showed `0 completed`, `0 pending`, and `0
exhausted`; no unsupported promotion or filtering metrics were rendered.

## Exhausted-job recovery follow-up

The fixture now accepts `--exhausted-job`, which creates one explicit write
and exhausts its five processing attempts before browser interaction. The
browser expanded **Recover failed processing**, loaded the scoped exhausted
job, and showed its attempt count and `ConnectionError` category. Clicking
Retry removed it from the exhausted list and reported pending processing.
Running consolidation completed one job and increased the fact count from
one to two. The recovered convention appeared exactly once. This is an
isolated control/persistence check using unavailable cognition and simple
embeddings, not extraction-quality evidence.

The browser also exposed missing FastEmbed selection and stale embedded-preset
guidance. The form now includes FastEmbed, labels unconfigured mock cognition
as disabled in the service, and removes the managed-service implication from
the embedded recall setting. Long diagnostic paths wrap, and memory cards no
longer stretch to the height of the evidence panel.

Evidence: [recovered job](../../../output/playwright/ui-job-recovery.png) and
[updated form and layout](../../../output/playwright/ui-job-recovery-fixed.png).

## Legacy, routing, and source review follow-up

The `--review-queues` fixture exercised the read-only legacy list and the
metadata-only unresolved-routing queue. The browser displayed the unassigned
legacy fact separately from the one scoped fact. Its HTML remained literal
text. The configured profile saw one routing record; the foreign profile's
record remained hidden. Dismissal removed the selected record, and explicitly
selecting the unconfigured-profile view exposed its separate diagnostic.

The processed source episode had expired in this fixture. The detail panel
showed `Unavailable locally`, while the retained evidence excerpt and source
identifier remained readable. No transcript was reopened to infer availability.

This check exposed overflow in the memory detail panel. After the layout fix,
the browser measured zero document overflow, a 363-pixel detail width and
scroll width, and a recall input spanning its 364.72-pixel container. The
HTML execution marker remained false. The console contained a favicon 404
and the existing password-field-in-form diagnostic, with no application
exception. The browser and temporary fixture were then stopped.

Evidence: [initial review layout](../../../output/playwright/ui-review-queues.png),
[intermediate layout](../../../output/playwright/ui-review-queues-fixed.png), and
[final wrapped evidence](../../../output/playwright/ui-review-queues-final.png).
