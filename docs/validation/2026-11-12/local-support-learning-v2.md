# Local support learning result

The frozen full-encoder attempt completed, but failed the semantic gates.
On all 31 tuning and 29 development cases, the initial model, final master
and deployed artifact each selected exactly the last candidate. All six
permutation groups in each reserved partition remained inconsistent. These
are observed output patterns; the run does not prove their internal cause.
The candidate is inactive and this configuration is closed.

Source is `22bbc32fd815920cf8a8f1ed65f8cd24223e7e21`. The unchanged
[recipe](../../plans/2026-11-12-local-support-learning-recipe-v1.md) trained on
56 cases and 138 targets across eleven connected groups. Tuning/development
contain 31/29 known cases in six groups each, with provisional ancestry review.
These are development results, not independent fresh acceptance.

## Results

The fresh float16 artifact and final float32-master/autocast reference made
identical decisions on every comparison frame.

| Partition | Precision | Supported retention | Answerable recall | Critical retention | No-answer abstention |
| --- | --- | --- | --- | --- | --- |
| Train, 56 cases | 31/74 = 41.89% | 31/40 = 77.50% | 31/38 = 81.58% | 9/9 = 100% | 0/18 = 0% |
| Tuning, 31 cases | 11/31 = 35.48% | 11/27 = 40.74% | 11/26 = 42.31% | 4/12 = 33.33% | 0/5 = 0% |
| Development, 29 cases | 9/29 = 31.03% | 9/24 = 37.50% | 9/24 = 37.50% | 4/8 = 50% | 0/5 = 0% |

Original gates remain precision and critical retention >=95%, answerable
recall and abstention >=90%; this diagnostic also requires supported retention
>=95% and no unavailable observations. All 116 known cases were assessed.
Critical denominators are supported opportunities, not independent facts:
the partitions contain three, seven and four distinct critical facts.

Tuning and development selections did not change from initialization. On
training, supported selections rose from 19 to 31, but false selections rose
from 37 to 43 and exact cohorts fell from 17/56 to 13/56. Seventeen training
cohorts added selections while keeping the last candidate. This is not useful
generalization. Development false support consists of ten wrong-subject or
relation selections, five unresolved-condition selections, four incomplete
source-support selections and one unresolved-conflict selection. Every error,
critical outcome and permutation group remains in the
[receipt](local-support-learning-v2-receipt.json), with
[raw predictions](local-support-learning-v2-predictions.json).

The eleven repeated mechanical boundary controls retain all 38 positive
opportunities. Seven were selected; four unavailable requests account for
17 opportunities, and answerable recall is 6/10. None is counted as an
independent semantic example or removed from the denominator.

## Computation and export

All 84 fixed updates and 336 microbatches completed. Every update passed
finite gradients and Adam-state checks, positive embedding/layer/head activity
before and after clipping, and unchanged scaler 128. The backbone and head
changed. Training recorded 336 backbone entries and 672 entries in every one
of the 22 layers, including checkpoint recomputation.

Export parity passed all 123 complete frames: maximum absolute logit error
0.01806640625, zero threshold/selection flips, matching target orders and frame
hashes. The saved float16 configuration reloads with dtype auto. The final
float32 checkpoint and inactive deployment weights are retained privately and
hash-verified. The archive contains their receipts, not the large weight files.

The child completed in 92.38 seconds, the owned job in 94.24 seconds and the
controller in 96.03 seconds. Peak allocated/reserved PyTorch memory was
2,787,564,544 / 3,005,218,816 bytes under the unchanged 10 GiB allocator cap.
This is not total-process VRAM accounting. All 103 frozen bindings and 25
runtime versions stayed unchanged; all five owned processes stopped. No
trained complete retrieval latency was measured.

## Preserved interruption and reviewed repair

The first corpus execution at `88bbcabb` failed during progress-file atomic
replacement with Windows access denied. It completed 62 updates and 251
microbatches; update 63 is partial and 85 microbatches remain unfinished.
It saved no final weights or trained evaluation. A concurrent reader without
delete sharing is plausible, but the locking process was not established.
The complete failed evidence archive is preserved at SHA-256
`625ba65bf3f3565ef045c809972de13da7672d21471f8a17af73e1e90d91cd98`.

The [prospective amendment](../../plans/2026-11-12-local-support-corpus-run-v1.md)
allowed one reviewed infrastructure-only restart. Its writer retries only
PermissionError, with a maximum total scheduled wait of 310 ms per write;
a persistent lock still fails. Tests reproduced a real Windows reader lock,
then verified the bounded fix. All thirteen contracts executed across the
controller/model environments. No original numerical setting changed. Six
prepared artifacts are byte-identical, including the entire schedule and all
123 frames. The completed attempt restarted from the original checkpoint and
seed, not partial update 63. The prior tokenizer-import failure and its repair
remain separately preserved in the backward-gate report.

Independent review reconstructed schedules, predictions, grades, parity,
artifact hashes and cleanup without discrepancies. The readback-verified
completed-run archive contains 111 members and 2,238,721 bytes, SHA-256
`568cf0fd3f208ede61991a0dcbbe5759dd247f8df5d1ca00edc98f28cfbd105f`.

## Product and goal state

Local operation remains required. The existing builder keeps judged paths
off by default, supports local Ollama judgments, and builds hosted Jev only
when explicitly selected as `typesafe` with its configured key. It never
switches to Jev after a local failure. The workspace key did not enable or
participate in this run; no hosted requests occurred.

This failed candidate does not justify activation, trained timing work,
fresh acceptance, another automatic run or a threshold/model/data sweep.
Existing local memory operation and the separately activated optional pilot
remain unchanged. There is still no passing combined quality/timing candidate
and zero qualifying original-pilot days. Original capture, native, two-week
pilot, package publication and delivery obligations remain open; the six-part
goal is active and incomplete.
