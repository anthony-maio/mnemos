# Native Claude plugin checkpoint

Source commit `6600dc262927661d61ce9962c812a022014281c7` and wheel SHA256 `349737206735f15b59562f3c7d53af51464176473b5de1fa12d2783cf039a25a` are bound in both native reports and the adapter timing report. The adapter timing gate passed.

The v9 native run is archived with terminal status `failed`, native return code `1`, and elapsed time `8914.0 ms`. The stream reported cost `0.365532` USD. Its run retained the host's account connector metadata in its report. Its max-budget result stopped before the requested full native lifecycle; SessionStart and UserPromptSubmit were observed while Stop remained unobserved.

The v10 native run is archived with terminal status `not_observed`, native return code `0`, and elapsed time `10567.2 ms`. The stream reported cost `0.029099999999999997` USD. It used an isolated process with account MCP servers disabled. The report records seeded recall, all three plugin hooks, durable restart, and the isolated plugin tool inventory, but no native Mnemos tool call was observed. The exact two native capture gate remains failed because the run recorded three captures, including a provisional event. The reconstructed plugin-hook replay uses retained transcript data and a synthetic hook payload; it is labeled reconstructed and is not the original native hook payload.

False criteria in v9: not reached before budget failure.

False criteria in v10: exactly_two_native_captures, user_transcript_association.

Reports preserve exact criteria, provider cost fields, hook/MCP telemetry, and digests. Helper snapshots are stored as `.py.txt` files. Private Claude fixtures, full native stdout/transcripts, authentication material, and account connector contents are excluded; These results do not establish native write lifecycle success, real-model quality, hosted behavior, or human acceptance.
