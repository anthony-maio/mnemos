# Local operation with optional Jev

The user's November 11 direction is local operation with an explicit Jev option.
The existing builder already supports local Ollama judgments and optional hosted
TypeSafe judgments. Keep judgments off unless selected, preserve the local
memory workflow, and never switch to hosted inference after a local failure.
The key is present in the workspace's private `.env`; its presence does not
enable hosted operation. Do not change live configuration or pilot installs.

Run one bounded Jev development comparison using the 44 hybrid-arm requests
from the frozen Qwen3.5 comparison. Preserve the exact questions, candidates,
sources, reference dates and 0.8 selection rule. Labels remain outside inference.
One unrelated warmup resolves the vendor model identity for the remaining calls.
Use verified HTTPS directly, no proxy environment, no redirects, one request
slot, no retries, 15 seconds per operation and 720 seconds overall. Record exact
request/response bodies without authentication headers. Stop on the first error.
Exclusive output creation prevents overwriting an attempt. No local model run,
new retrieval, extraction, fresh acceptance or live memory use is included.

This establishes whether the existing optional backend helps this known
development set. It cannot establish that local semantic support works, measure
critical retention, pass complete retrieval timing, or replace fresh acceptance
and the actual pilot. Keep the original gates and all failed results intact.
Do not respond to a failure with another prompt, threshold or model sweep.

Official references checked November 11:
[API](https://docs.typesafe.ai/api/system-one),
[model identities](https://docs.typesafe.ai/api/list-models),
[pricing](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost).
The published price is $0.042 per million input tokens with output tokens free.
The comparison refuses more than one million estimated input tokens; actual
reported usage and estimated cost must accompany the result.
