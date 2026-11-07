# Forget selection and native catalog preparation

Product `6d0e5ba6` clarifies how to review advisory wording and sources that
retain requested information without a derived fact. Deferred captures remain
eligible when the preview says so. Candidate requirements are a minimum source
dependency set. The extraction-review tools now identify their separate purpose.
The changes leave selections explicit and preserve the storage deletion guards.

The native runner reads the installed catalog, denies tools outside each phase,
and requires an exact match between the phase allowlist and the native startup
catalog. This addresses v4's eighteen advertised tools versus four permitted
tools. It does not authorize the denied review request or reclassify v4.
The strict parser, original inspection, dependency, deletion and replay checks
remain in force.

All 180 focused checks pass, including fourteen new native-boundary cases.
Nine initial cases failed before the repair. The first broader run had 179
passes and one failure because the rewritten preview description omitted its
read-only designation; that designation was restored before the passing run.
Formatting passes for six changed Python files, and strict types pass for
four source modules. The [preparation receipt](review-selection-v5-preparation.json)
preserves these results and the private archive.

The saved v4 preview still reconstructs exactly through the independent verifier:
five pages, eighteen candidates and twenty-two additional sources. The largest
wrapped page is 11,961 bytes. The [rendering receipt](review-selection-v5-rendering.json)
records this offline measurement. It does not establish improved model selection
or token efficiency.

Wheel/source builds, Twine and a fresh [installed CLI/MCP smoke check](installed-smoke-6d0e5ba6.json)
pass. Wheel SHA-256 is
`ecd11eb7ecbb9cc4f14552d9a7253dead7a8348879c35063411bb850abe06036`.
The fresh native environment matches all 57 preceding package versions and
exposes eighteen installed tools. That catalog is installation evidence; the
native session still needs to demonstrate phase filtering.

The [v5 protocol](../../plans/2026-11-06-native-claude-review-selection-v5.md)
keeps the prompts, scenario, host/model, budgets and single-attempt limit.
Full regression and native validation have not started at this preparation
checkpoint. Require the committed runner and a passing frozen full suite before
inference. The pilot remains unstarted and live configuration is unchanged.
