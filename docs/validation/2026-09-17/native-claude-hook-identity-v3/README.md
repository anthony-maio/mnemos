# Native Claude hook-time identity observation

Claude Code 2.1.191 completed one fixed synthetic prompt with
claude-sonnet-4-6 in 8.962 seconds. Reported cost was $0.003708, and its
stream contained no tool calls. The disposable service used the installed
6600dc2 package, constant vectors, unavailable cognition, and private paths.
Copied authentication was removed and the service descriptor was cleaned up.

The wrapper observed the real hook payload and transcript immediately before
and after delegating to the unchanged installed adapter. UserPromptSubmit
contained no prompt ID, and its transcript file did not yet exist at either
observation. By Stop, the transcript contained six rows and a user record
with both UUID and promptId. Stop supplies no prompt text, so its false
exact_prompt_match field does not mean the transcript text mismatched.

This rules out reading a preexisting native transcript identity during the
prompt hook for this observed turn. It does not justify text-only association
or removal of the capture warning. Both capture hooks still emitted diagnostic
stderr. This is timing and identity evidence, not successful capture-health,
fact-lifecycle, extraction-quality, or cross-host acceptance.

The report retains the v2 fixture/profile prefix used by the v3 helper. The
executed helper hash and native byte hashes bind the observation. Raw native
streams remain in the private fixture; this archive contains metadata only.
