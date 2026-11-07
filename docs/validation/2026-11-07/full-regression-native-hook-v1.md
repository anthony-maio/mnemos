# Full regression after the Windows transport changes

The complete suite passes at `9c84909f`, which contains product source
`cddf8adc`: 3,495 passed, four skipped, zero failures and zero errors across
3,499 collected tests. Pytest reported 615.92 seconds. The separate JUnit suite
time was 613.014 seconds; the receipt labels both values explicitly.

The four skips cover two real WSL Hermes integration checks without the required
interpreter selection and two POSIX hook-command checks that require a POSIX
executable and shell. They are not runtime passes on those platforms. The
Windows transport checks ran without skips.

The runner selected the canonical website file explicitly and used a new test
temporary directory. Git HEAD and tracked source stayed unchanged. All 208
historical fixture files and the website file retained their original hashes.
An independent audit reconstructed the test counts from JUnit, rechecked those
bindings and confirmed that both recorded test processes had exited.

The verified private archive contains ten members, including the original
pytest output, JUnit results, controller record, audit and fixture manifest.
SHA-256:
`0d08d24786c341afa029800270b42d2c80e99ea2ed70175ef792c21f572036be`.
The archived audit's `pytest_wall_seconds` field contains the JUnit suite time;
the public receipt distinguishes it from the console summary. The raw results
and both durations are unchanged.

The [separately verified wheel](native-hook-portability-v1-receipt.json) matches
the current product source and passed an installed Windows request. The
[complete performance comparison](native-hook-performance-v1.md) remains bound
to its earlier measured wheel: Stop improved, prompt submission did not
consistently improve, and both paths still fail the 100 ms requirement.

This result closes the current source-wide regression check. Current native
Codex/Claude acceptance, fresh answer-support quality, remaining context work,
canonical integration and the actual two-week private pilot remain open. The
pilot has zero days; no live configuration or publication changed.

- [Verification receipt](full-regression-native-hook-v1-receipt.json)
- [Improvement plan](../../plans/2026-11-06-mnemos-improvements.md)
