# Generated UUID metadata rejection

A deterministic probe reproduced a rejected explicit remember operation with generated UUID `abcdefab-1234-4567-8123-123456789012`. The write firewall redacted its numeric tail as a credit-card number, so source metadata validation rejected the operation before storing any event, job, or fact. A UUID with alphabetic groups passed and produced one durable fact after maintenance.

The probe temporarily substituted the UUID generator in its own process; it did not change production source or live data. The service and firewall source match commit `7443582`.

The full context-authority validation had an empty-fact failure at the initial remember step, but its generated ID was not retained. This probe confirms a reachable write defect without establishing that specific failure's cause. The defect remains open at this checkpoint.
