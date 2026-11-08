# Mnemos ClawHub Skill

The repo ships a ClawHub skill source at `skills/mnemos-memory`.

What it does:

- teaches OpenClaw agents how to install `mnemos-memory[mcp]`
- tells them to run `mnemos ui` to create canonical config
- explains how to wire `mnemos-mcp` into the current host with `MNEMOS_CONFIG_PATH`
- reinforces the `retrieve -> work -> store -> consolidate` loop
- keeps host claims accurate by not promising hard auto-capture outside verified hook-capable hosts

This checkout is an unpublished 0.6.1rc1 candidate. No ClawHub upload is prepared
or authorized by package validation. Reconcile the skill's version and current
host claims with the final reviewed release before publishing it.

Positioning:

- This skill is an onboarding and usage layer for OpenClaw agents.
- It is not a guarantee that every OpenClaw host exposes lifecycle hooks for automatic capture.
