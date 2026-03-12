# Mnemos ClawHub Skill

The repo ships a publishable ClawHub skill at `skills/mnemos-memory`.

What it does:

- teaches OpenClaw agents how to install `mnemos-memory[mcp]`
- tells them to run `mnemos ui` to create canonical config
- explains how to wire `mnemos-mcp` into the current host with `MNEMOS_CONFIG_PATH`
- reinforces the `retrieve -> work -> store -> consolidate` loop
- keeps host claims accurate by not promising hard auto-capture outside verified hook-capable hosts

Publish command:

```bash
clawhub publish ./skills/mnemos-memory --slug mnemos-memory --name "Mnemos Memory" --version 0.3.0 --tags latest
```

Positioning:

- This skill is an onboarding and usage layer for OpenClaw agents.
- It is not a guarantee that every OpenClaw host exposes lifecycle hooks for automatic capture.
