# Recovery extraction control preparation

Two provider-only calls, one per original saved user/assistant event, using the original d937892 wheel code and no existing facts. Requests are reconstructed, not recovered wire prompts. The no-store guard rejects any storage access. Qwen3.5:9b digest is checked against the earlier recovery run. Only thinking is requested false; this also changes the structured route from chat to generate. Temperature 0, 8192 tokens and 180-second timeout are unchanged. No database is opened and no production setting changes.
