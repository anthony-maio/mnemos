# Disposable retained-weight backward probe

Before corpus training, run one synthetic forward/backward/AdamW update against
the bound retained checkpoint's eight-layer prefix and rank-8 q/v adapters.
Use five targets and exactly 552 tokens, the current corpus maximum. This is
a training-memory check, not a new inference context bound; the 4096-token
deployment limit and completed capacity result remain unchanged. No corpus
inputs, labels, tuning decisions or development outcomes enter this probe.

Use seed 2026110808, float16 original parameters and native SDPA, float32
trainable parameters and BCE loss, and alternating synthetic labels 0/1/0/1/0.
Use the proposed AdamW groups, clip unscaled norm at 1.0 and freeze GradScaler
at init_scale 128, growth_factor 2, backoff_factor 0.5, growth_interval 2000.
Freeze source, model, tokenizer and request hashes before loading weights.

Require finite loss, every expected trainable gradient and every updated
parameter; exactly one optimizer update per trainable parameter; unchanged
frozen-parameter hashes; changed trainable hashes; no gradients on frozen
parameters; and unchanged file bindings. Record peak CUDA allocation/reservation.
An exception, nonfinite value or skipped update fails the probe. No parameter
or optimizer checkpoint is saved. One exclusive destination prevents restart.
Retain failure evidence and make a new reviewed decision before any retry.

This probe cannot establish semantic quality, convergence, full-corpus gradient
stability, merged-adapter numerical agreement or complete recall/capture latency.
The exact training/baseline execution freeze still precedes corpus training.
