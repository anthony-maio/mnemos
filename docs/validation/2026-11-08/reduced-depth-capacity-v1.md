# Eight-layer support capacity result

The fixed-depth candidate passes its necessary computation-only screen. Source
was committed before execution at `6e31d471` (implementation `8cf27dba`). The
original full-depth and portable-attention failures remain closed. This result
does not establish semantic quality or complete retrieval latency.

The model retains the original embeddings, first eight of 28 blocks, rotary
encoding and final normalization, with a random shared target head. Native
SDPA and float16 are unchanged. The run used the bound local checkpoint,
torch 2.10.0+cu128, transformers 4.57.6 and RTX 4080. Full checkpoint loading
reported no missing, unexpected, mismatched or erroneous keys before pruning.

Review found no blocking defect and required exact workload parity before
execution. Both requests.json and frames.json were byte-identical to the
original joint-support-capacity-v2 artifacts. Their SHA256 values are
`22ebaca81e628b1ac294812f1ad090fbf09357c76b275da6a80d375775d5da09` and
`3bf8729c7345e231333c6d6f90a50cf41b10488f4622d713be3e281bbe00dc39`.
The 4096-token and 64000-byte limits, source ownership, target ordering and
overflow rejection were preserved.

| Shape | Measured samples | Median ms | p95 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| Empty | 50 | 0.39 | 0.56 | 0.60 |
| One short target | 50 | 15.02 | 19.15 | 29.38 |
| Five short targets | 50 | 15.48 | 19.67 | 29.40 |
| 4096 tokens, one source per target | 50 | 231.92 | 241.01 | 251.03 |
| 4096 tokens, eight sources per target | 50 | 232.29 | 237.40 | 259.84 |
| 64000-byte payload | 50 | 44.04 | 48.70 | 72.35 |

One execution completed all 330 samples: five warmups and fifty measured
repetitions for each of six shapes. Timing includes serialization, transfers,
the backbone and shared head, output validation and synchronized completion.
All frozen file bindings remained unchanged. These are single-run warm timings;
there is no installed retrieval, 10000-fact store or trained classifier in this
screen. The approximate 250 ms planning estimate was not used as a pass gate.

Five tiny random-model structural checks and five unchanged shared-head tests
passed. Four harness checks passed for terminal exceptions, interruption sample
retention and restart prevention. These establish mechanics, not model quality.
The reviewer inspected code and did not execute tests or model weights.

Private evidence remains in `.venv/reduced-depth-capacity-v1`. Freeze SHA256 is
`7d1f98d3768f801ab70c2d49cb2364df7d48c2b20867cb0323ea18568546c2ba`;
result SHA256 is
`28d4537b95b669d8ebba8218acf730b07a490914170e64d6d8a6b4167a77c428`.
The readback-verified archive `.venv/reduced-depth-capacity-evidence-v1.zip`
contains 15 files plus a manifest, SHA256
`178f7b4388c141a59f03c4b699fc83bc595dab1bce44f2f98422812af5f16e16`.

Next: prepare and review a fixed learning recipe against the existing grouped
corpus, including imbalance, sparse condition categories, critical-fact limits,
training memory, checkpoint selection and threshold selection. No training has
started and no fresh acceptance cases were consumed. The original semantic,
complete-latency, current-artifact native, publication and pilot gates remain.
