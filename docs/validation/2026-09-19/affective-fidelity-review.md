# Affective classifier fidelity review

This is a retrospective review of the archived mechanism evaluation. It does
not rerun the model and does not alter the 300 original observations.

## What was measured

The source is `mechanism-heldout/module-mechanism.json` inside
[`phase-artifacts.zip`](mechanism-local-v1/phase-artifacts.zip). Its
`affective.summary` reports 50 pairs, 100 complete conditions, 300
observations, and three repetitions per condition. Every call was parsed:
`classifier_calls=300`, `classifier_parsed_states=300`, and
`classifier_fidelity_missing=0`.

The evaluator records `classifier_distance` in
[`module_mechanism.py`](../../../benchmarks/local_memory/module_mechanism.py)
using `classifier_state.distance(condition.state)`. The frozen protocol calls
this normalized Euclidean distance and pre-registers `<= 0.2` as a fidelity
pass ([protocol](../../plans/2026-09-16-module-evaluation-protocol.md)).
The archive summary therefore gives 33/300 fidelity passes and 267/300
failures. Those 33 passes are 11 urgent conditions repeated three times, not
33 independent cases. Distances range from 0.0408248 to 0.3511885, with mean
0.2540342 and median 0.2198484.

## The axis pattern

The held-out dataset is
[`module-mechanism-v1.json`](../../../benchmarks/local_memory/datasets/module-mechanism-v1.json).
Its authored condition states contain two repeated gold vectors:

- Calm: valence 0.4, arousal 0.1, complexity 0.8.
- Urgent: valence -0.6, arousal 0.9, complexity 0.3.

Against those values, the 150 calm observations produced valence 0.8 in all
cases; arousal 0.4 in 120 and 0.6 in 30; complexity 0.6 in 120 and 0.9 in
30. None passed the distance criterion.

The 150 urgent observations produced valence 0.2 in 84, -0.4 in 30, 0.1 in
21, -0.2 in 12, and -0.6 in 3; arousal 0.8 in 78 and 0.9 in 72; complexity
0.6 in 87 and 0.4 in 63. Thirty-three urgent observations passed. The
repeated values show a systematic axis mismatch in this archived run, rather
than a response parsing failure.

## What the result supports

The result supports a classifier-fidelity failure against the frozen authored
fixture. It does not establish an end-to-end affective retrieval benefit.
Ranking uses the supplied frozen condition state in `evaluate_affective_pair`,
while classifier fidelity is measured separately. The archive report states
that the ranking result is based on synthetic gold states, and its summary
records `gold_states_are_synthetic_labels=true`.

The semantic query embedding is held constant within each pair while the
condition wording carries the calm or urgent state. This makes the ranking
comparison controlled, but it does not independently validate that the numeric
gold vectors are the right interpretation of those words. There is no human
annotation or external calibration in this archive that would justify treating
the authored values as ground truth for real affective judgment.

The separate [response parsing repair report](affective-response-contract-v1/README.md)
replayed all 300 archived responses and found zero changed states. It therefore
does not explain, and did not improve, the original 33/300 fidelity result.

## Next development step

Independently review and validate the affective fixture: define the axis
semantics, obtain an independent annotation or calibration basis for the gold
states, and preserve a reviewed fixture before rerunning the unchanged fidelity
measurement. This review does not support changing the threshold, tuning the
prompt, or making production changes.
