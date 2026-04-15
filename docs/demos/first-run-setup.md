# First-Run Setup Walkthrough

This is the intended `0.7.0` onboarding path for new Mnemos users.

## Goal

Get from install to a trusted local setup without hand-editing MCP config files or environment variables.

## Flow

1. Install Mnemos with MCP support.

   ```bash
   pip install "mnemos-memory[mcp]"
   ```

2. Launch the control plane.

   ```bash
   mnemos ui
   ```

3. Choose **Starter SQLite**.

   Use this as the default local-first path:

   - one SQLite file
   - guided host setup
   - smoke-check verification
   - inspectable local memory

4. If your sessions are noisy or longer-running, switch to **Stable Memory**.

   That preset keeps the same SQLite path and enables recall-gated consolidation so only better-supported facts are promoted into long-term memory.

5. Apply the host integration from the UI.

   - Claude Code is the primary path
   - Cursor and Codex stay documented but secondary in `0.7.0`

6. Run the built-in smoke check.

   A successful first run should make these states visible in the home screen:

   - setup status
   - selected host and support tier
   - visible chunk count
   - last smoke-check result
   - next actions if the setup is still degraded

## What “done” looks like

You should be able to leave the first session with:

- a saved global `mnemos.toml`
- a working SQLite path
- a host integration applied from the UI
- a passing smoke check
- clear next steps if you are still on mock/simple providers
