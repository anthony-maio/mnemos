# Held-out generation settings

The lifecycle launcher now accepts temperature, output-token limit and timeout
through its API and CLI. Defaults remain 0.0, 512 and 120 seconds. Existing
Ollama settings validation runs before discovery or output creation. The same
values reach the service factory, manifest and options hash. The harness
already rejects checkpoint resumes with changed options hashes. Use new output
and work directories for a different configuration.

Validation on the implementation checkout:

- Ten new tests failed before implementation because the API/CLI did not accept
  the settings. Invalid values, factory wiring, manifest/hash changes and CLI
  forwarding are covered.
- `python -m pytest -q tests/test_heldout_generation_settings.py tests/test_benchmark_thinking_option.py tests/test_evaluation_registry_identity.py tests/test_local_memory_benchmark.py`: 75 passed in 277.04 seconds.
- Black check passed for the changed launcher and new test file.
- Python 3.10 mypy passed for the changed launcher.
- Independent read-only review found no correctness defect and independently
  ran the ten new tests successfully.

No real-model call was made by these tests. This is not a new full-suite or
installed-package checkpoint. The prior full checkpoint remains c1e2217 with
2,081 passes and four skips. No production engine, default model, scoring
threshold or historical artifact changed. The completed 14B diagnostic used
its own previously frozen runner and source, not this launcher change.
