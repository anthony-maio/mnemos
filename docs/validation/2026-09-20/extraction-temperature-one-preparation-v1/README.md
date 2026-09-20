# Temperature-only extraction diagnostic: frozen preparation

Status at preparation: NOT RUN. This freezes 40 calls over all 20 previously
executed authored cases, using the original production/history prompt arms.
Only temperature changes from zero to 1.0. Source 05c71d2, model digest,
8,192-token budget, thinking mode, schema and rubric remain fixed. The recent
missing-target prompt clause is not included. No selective retry is allowed.

The [model author's guidance](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)
recommends nonzero sampling temperatures, including 1.0 for thinking general
tasks, alongside other sampling controls. This experiment isolates temperature;
it does not implement that entire recipe or establish why earlier calls failed.
The local server reports Ollama 0.34.1. Existing reports about older releases
are not proof of a defect in this installation.

One stochastic repetition can identify a candidate for further testing but
cannot establish reliability or justify adopting a production configuration.
All runtime errors, omissions and unsupported assertions remain failures under
the original authored rubric. Earlier results and P07/N10 ambiguities remain
unchanged. These are reused development cases, not unseen held-out evidence.
The terminal result belongs in a separate archive after the run finishes.
