# Installed stdin-controlled owner check

The v5 installation completed all six captures under the original stdin-controlled owner setup. Both JSON and SQLite owners kept their main thread reading the driver's stdin pipe while the worker handled native Stop requests. The unchanged native v3 launcher used the original 30-second timeout. Each arm stored three events and completed three jobs, with matching original identities and payloads across each pair.

This is the setup in which v4 timed out on the first JSON warmup. V5 differs by isolating the Git discovery subprocess from the owner's stdin. The successful reproduction check supports that fix for the observed scenario. The separate process-level regression establishes that discovery cannot consume host input. Neither result establishes every possible cause of Windows subprocess stalls.

An independent audit reopened both stores, matched all six event/job records and response hashes, checked both queue formats empty, verified installed file hashes, and rechecked all owner and launcher processes absent and all four ports closed. Both owners exited zero with graceful stop records. The runner also verified the native launcher unchanged.

The 17-member evidence archive includes the frozen protocol, helpers, input manifest, raw records, response receipts, owner logs and shutdown records. SHA256: 9cc0b14b50e9197b5a2bac025eefcc38c3ff8189f3ed03a1b5c008136c8b13bf. The adjacent JSON receipt records the wheel, source commit and evidence hashes.

Three pairs are a bounded reproduction check, not a new performance comparison. No latency percentile or 100 ms acceptance is claimed. The previous failures and the complete v4 timing comparison remain unchanged. Live pilot configuration, memory databases and canonical delivery are unchanged. Quality, full retrieval timing, fresh acceptance and actual daily-use obligations remain open.
