# Extraction contract development v2

Four candidate-only calls used the same fixed development cases, gemma3:4b digest a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a, temperature 0, and 512 output tokens. Candidate a27ad6e has production code SHA256 ebd2e00e0508d12df9b60562d2f9adc841dd0bea1699e5bd187842a3dfc45b5a. The baseline was not rerun. This is development evidence, not held-out acceptance.

All four responses passed their request schema and runtime validation. The no-facts decisions now included required explanations. All returned proposals cited current source events. No facts were persisted.

Only the question case passed the frozen exact contract proxy (1/4). The support case emitted add; the correction case emitted add without a target/version; the new-preference case incorrectly declined to extract a durable preference. These operation and omission errors remain material despite valid output structure. No semantic precision claim follows from this sample. The schema repair enforces evidence/target/version constraints when an operation is selected; it cannot force correct operation selection.
