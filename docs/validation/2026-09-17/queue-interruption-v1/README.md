# Interrupted evaluation queue snapshot

This is a read-only snapshot of the interrupted real-model-v4 evaluation queue. It records the queue status and coordinator lock byte-for-byte, hashes completed-phase artifacts and queue output files, and records only checkpoint counts. It does not retire the lock, resume the queue, inspect quality results, or call a model or network endpoint.

The frozen source pin was `d94ded67a6b3fe6b6f7474a74518d4efb143698e` and the queue runner SHA256 was `2e642e3b4bb7e595d9c676ec51780b8cbe4a1e6449b29fee3a57d1307d79f84e`. The checkpoint contained 62 completed entries, 112 graph rows, and 150 affective rows at capture time. Raw queue outputs remain in their original location; this archive stores their relative paths, byte counts, and SHA256 values only.

The coordinator and phase process IDs checked by this snapshot were not alive at either liveness check. The parent observed an Ollama tags timeout during the attempted recovery separately; this snapshot did not reproduce or test that endpoint failure. At snapshot time no recovery had occurred and the stale lock was unchanged. The subsequent [recovery record](../queue-recovery-v1/README.md) documents the verified resume at 18:29 UTC.

This evidence establishes interruption state and file preservation only. It does not establish completed held-out quality, model behavior, or a safe resume.
