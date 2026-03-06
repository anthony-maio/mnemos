---
name: mnemos-prime
description: Prime biomimetic memory for the current task by retrieving relevant prior context and setting storage/consolidation behavior.
---

# Prime Mnemos Memory

Use Mnemos memory intentionally for this session.

## Steps

1. Call `mnemos_retrieve` with a focused query based on the user's current request.
2. Summarize top relevant memories in 3-6 concise bullets.
3. Continue the task while storing only important updates:
   - facts about the user's stack/preferences/goals
   - architectural decisions
   - fixes for recurring errors
4. Before finishing substantial work, call `mnemos_consolidate`.

## Guardrails

- Do not store every turn.
- Prioritize high-salience, durable facts.
- If retrieval returns nothing useful, proceed normally and seed memory as you learn.
