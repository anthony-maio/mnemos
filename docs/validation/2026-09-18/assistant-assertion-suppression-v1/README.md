# Assistant assertion suppression

Source based on `0d1de4591b516f1ae554386f94b3ec549473cc4c` plus `source.patch` and the archived new test file passed 1,030 tests with 4 skips, Black, strict mypy, and JavaScript syntax. The report hashes all Python source and test inputs, UI JavaScript, and project configuration before and after validation. Temporary test fixtures were outside Git on C:.

Automatic extraction previously checked a generated role-prefixed assertion against forget suppressions for the unprefixed fact. It now checks the sanitized assertion before adding the role prefix. Atomic commit repeats that check using the persisted source-event role, so candidate metadata cannot bypass it. Scope and profile stay part of the canonical key. Support proposals retain the existing target content.

The 8 regression cases cover queued assistant extraction after forgetting, atomic commit with forged role metadata, independent project and profile isolation, an unrelated proposal in the same batch, explicit reteaching, support preservation, and the non-exact paraphrase boundary. Provider responses are deterministic fixtures. This establishes exact normalized suppression, not semantic topic erasure.

No installed package, native host acceptance, or real-model quality claim is made for this source checkpoint. Native v7/v8 used the earlier ac139dd package. The frozen Ollama evaluation was not changed.
