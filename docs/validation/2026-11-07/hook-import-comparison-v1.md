# Installed wheel comparison after the import repair

Separating retrieval settings removes the unused domain models from hook
startup. The measured gain is small relative to the complete hook target.
Across ten alternating-order pairs, the median paired empty-hook difference
is -24.12 ms. One candidate sample is much slower. Settings import has no
clear practical improvement in this run.

| Case | Baseline median ms | Candidate median ms | Median paired difference ms | Baseline maximum ms | Candidate maximum ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Settings import | 446.89 | 450.44 | -2.56 | 703.50 | 543.98 |
| Empty hook | 707.54 | 689.58 | -24.12 | 1069.33 | 1457.20 |

The paired difference is candidate minus baseline within each round, so it
need not equal the difference between the two arm medians. Nine empty-hook
pairs favor the candidate; the remaining pair is 771.54 ms slower. All forty
observations, including that outlier, remain in the result. This small
diagnostic sample does not establish a general latency improvement or a p95
acceptance result.

Both wheels were installed into separate private directories on the same
volume using `pip --target`, with the same interpreter and dependencies. Both
copies were byte-compiled before timing. Each subprocess uses the same isolated
bootstrap to add its installed directory, verify the loaded package and run
the unchanged launcher. No source checkout is imported. This adds the same
diagnostic bootstrap to both arms; it is not the original complete hook command.
The empty payload performs no capture, retrieval or service request.

The baseline is `749f1213`, wheel SHA-256
`2884c412f10ce130aed0afad0910c461ba945657f465f73ec604def78ab3ba2e`.
The candidate is `3e137634`, wheel SHA-256
`5c789fe32b3130f226c603ff17f313e4fe54b3e14652d1af2f4f012bb4df689b`.
Every installed package file matches its wheel, and every candidate Python
source matches the checkout. All candidate samples omit the retrieval domain
models; all baseline samples import them. Neither arm imports the engine,
FastEmbed or NumPy. No database or service runtime was created.

The candidate passes 37 configuration/retrieval checks and 68 adapter, client,
service and import checks, with three shared cases, plus 12 documentation
checks. Black, strict mypy, isolated wheel build and Twine pass. Complete
settings and retrieval JSON schemas match the baseline. The first build
without isolation lacked Hatchling; the standard isolated build succeeded with
Hatchling 1.33.0. Both build logs are retained.

The independent audit checked all forty observations, paired calculations and
191 file bindings. The verified private archive has 572 members, SHA-256
`7d2338230628cbaf46bf908dc017aa82da10c190a46a86f6f9c4c9d5553a4731`.

The full 100 ms hook requirement remains unmet by the original measured
configuration and has not been repeated for this candidate. The dependency
repair alone leaves about 690 ms of empty-command startup in this sample.
Further work must address the cost of starting and configuring a new process
for every event while preserving scope checks, write safety and durable outage
handling. No live configuration, pilot installation or published package changed.

- [Comparison receipt](hook-import-comparison-v1-receipt.json)
- [Startup and real-capture diagnosis](hook-startup-diagnostics-v1.md)
- [Original complete-command performance](installed-performance-v3.md)
