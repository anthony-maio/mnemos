# Capture and forget overlap v2

This offline diagnostic uses frozen source commit `9e253390c58302817a18b6504f55899b36a2c27b`, a disposable SQLite database, `UnavailableLLMProvider`, and `SimpleEmbeddingProvider`. No live database, model endpoint, host, or configuration was used. The runner is archived as `capture_forget_overlap_probe_v2.py.txt` so repository Python checks do not execute validation artifacts.

The exact fact `The project release channel is overlap-old-5c7a.` is embedded in the captured prompt before any service calls. The captured prompt is promoted into one fact by the all-modules-disabled ablation. A separate explicit remember is corrected to version 2 and forgotten. Retrieval of the old token still returns the captured prompt fact immediately after forget and after SQLite restart.

This is a real overlap result for the raw capture ablation. Product `ModuleConfig` defaults all five switches to true; retrieval profile `baseline` is a separate ranking choice. The result does not claim that default cognition promotes every prompt verbatim. The service's reported forget boundary remains `live_memory_and_replay_suppression`, and this diagnostic does not test paraphrase or topic-level deletion or historical backup erasure.

