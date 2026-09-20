# Affective response parsing repair

The classifier asks for three comma-separated floats. The former parser took
the first three digit sequences anywhere in a response. It interpreted
`-.3, .8, .6` as `(1, 1, 1)` after clamping, split scientific exponents into
separate values, and attached a measured state to prose containing unrelated
numbers. These are parser defects independent of classifier judgment.

The repaired parser accepts exactly three finite numeric values, including
signed decimals and scientific notation, and preserves the existing clamp
for finite values outside the configured ranges. It returns unavailable for
prose, extra or missing values, nonfinite numbers, and exponent overflow.
An invalid response leaves the previous state history intact and does not
attach a measured state to the new memory.

`report.json` compares the actual previous parser at `45f1b01` with the repaired
source, retaining both files and their hashes. Five independent development
examples show the changed behavior and the preserved clamping contract.
Replaying all 300 archived real-model classifier responses produces zero
changed states. No inference occurred. The original 33/300 fidelity result
remains unchanged; this repair cannot explain that quality failure.

The new contract tests exercise classification, state history, and memory
tagging through the public methods. The focused affective/module checks pass
61 cases. Full package verification remains a separate checkpoint.
