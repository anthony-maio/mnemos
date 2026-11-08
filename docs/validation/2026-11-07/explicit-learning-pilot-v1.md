# Optional explicit-memory learning pilot

Four private project patches are prepared for a possible Codex learning pilot.
They add a project-local MCP entry and explicit-use guidance to each of the two
selected projects. The proposal uses the existing inactive `f5a944ea` wheel,
local BGE embeddings and disabled experimental modules. Nothing is activated.

This mode is additional learning work. It cannot pass the original full-pilot
entry checks, establish semantic quality or start the original two-week pilot
clock. The failed quality and timing results remain unchanged. The user must
choose this separate scope before it becomes an active pilot.

The proposed behavior is narrow and explicit:

- Use memory only when the user requests remember, recall, inspect, correct or
  forget. No capture hooks, scheduled use or automatic task-start recall.
- Inspect each returned fact and retained source before relying on it. Check
  current repository evidence where the fact may have changed; missing or
  conflicting evidence remains an unresolved result.
- Store only the requested assertion. Follow its original durable job ID and
  inspect the resulting fact before reporting completion. Preserve reviewed
  correction/forget boundaries and unrelated controls.
- Expose twelve named tools. Capture and review-promotion tools are excluded.
  Codex documents `enabled_tools` as its MCP tool allow list in the
  [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

The pinned Codex binary parsed the exact server entry and retained the tool
list. All 96 installed package files match the frozen wheel. The rollback
rehearsal restored four disposable copies exactly and refused eight deliberate
drifts. All 14 live paths remained unchanged. These are preparation checks;
project-level loading, the tool filter and the workflow still need a native
check on this exact setup before activation. No model turn, service start or
authentication copy occurred during preparation.

Project memories and raw pilot records remain local and unpublished. Requested
facts and source excerpts can enter Codex's normal conversation/model context.
This proposal does not promise entirely offline processing. The private review
folder contains all four patches, their expected hashes, the exact instructions
and rollback plan. No user-wide server registration or Claude change is proposed.

The [receipt](explicit-learning-pilot-v1-receipt.json) records the archive and
checks. Scope approval and activation approval are both absent. The existing
normal-hook review request belongs to the original full-pilot path and remains
separate.
