# Independent Hermes candidate repair checks

The isolated host candidate now passes the six bounded checks exercised by
the preceding write and hybrid-copy audits. The exact patch hash is
`9f44eb5b193c0e773f962991670420140f59684b949f62ee2c805c4f76ac2d79`.

- Single-message user writes remain in the evidence projection.
- Tool responses save successfully and contribute no user/assistant evidence.
- Assistant tool calls with null content save and contribute no text evidence.
- Multimodal user text saves and projects its actual text with the native ID.
- An unchanged hybrid carrier copied through the real model-facing restore
  path preserves both canonical origins.
- Role/content tampering of a copied summary component is rejected.

The intermediate component-v2 rerun is retained: single-message capture and
null-content writes were fixed, but tool writes still failed and multimodal
text was still omitted. The earlier four original failures and hybrid-copy
identity failure remain in their own archives.

These are independent SessionDB and compressor fixture checks. They do not
establish every lifecycle caller, database upgrade compatibility, a complete
installed integration, actual model compression, or a live host upgrade.
Broader host suites and installed replay/suppression work remain separate.
No live host source, configuration, or memory data changed.

The evidence index binds 11 artifacts. This README is outside the index.
