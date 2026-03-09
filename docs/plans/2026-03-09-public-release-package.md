# Public Release Package Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship an honest, launch-ready public OSS package for Mnemos that improves onboarding, feedback collection, and release messaging without overstating production maturity.

**Architecture:** Keep the product surface narrow and explicit. Update the public docs to match the tested v1 posture, add GitHub intake templates so feedback lands in the right buckets, and publish one release package doc that tells maintainers what to ship and what not to claim yet.

**Tech Stack:** Markdown docs, GitHub issue/PR templates, existing Python project docs surface.

---

### Task 1: Refresh the public-facing README and changelog

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_docs_consistency.py`

**Step 1: Update stale README release sections**

- Replace remaining category-defining comparison copy with benchmark-backed, release-honest language.
- Make the support surface point to Claude Code, Claude Desktop, generic MCP, and documented Codex setup.
- Add links to OSS trust docs and release-readiness docs.

**Step 2: Refresh the changelog**

- Move the current release surface into an `Unreleased` section.
- Call out scope isolation, host validation, Codex docs, and OSS packaging work.

**Step 3: Verify docs consistency**

Run: `python -m pytest -q tests/test_docs_consistency.py`

Expected: PASS

### Task 2: Add GitHub intake templates for public OSS traffic

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/ISSUE_TEMPLATE/host_compatibility.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/pull_request_template.md`

**Step 1: Add issue forms**

- Make bug reports ask for `mnemos-cli doctor`, host, provider, scope, and repro steps.
- Make feature requests ask for user problem, workflow, and success criteria.
- Make host compatibility reports ask for client, transport, config, and observed behavior.

**Step 2: Add PR template**

- Require scope of change, tests run, docs updates, and compatibility impact notes.

**Step 3: Keep the templates aligned with current release claims**

- No template should imply team memory, hosted sync, or Codex Tier 1 support.

### Task 3: Publish the release package guide

**Files:**
- Create: `docs/public-release-package.md`
- Modify: `README.md`
- Modify: `SUPPORT.md`

**Step 1: Write the release package doc**

- Include release framing, approved claims, blocked claims, ship checklist, pilot ask, and feedback routing.

**Step 2: Link it from the repo surface**

- Add README links so users and maintainers can find the release package, compatibility matrix, security policy, support policy, and checklist quickly.

**Step 3: Tighten support messaging**

- Make `SUPPORT.md` point new users to the right diagnostics, docs, and issue form paths.

### Task 4: Run the public-surface verification pass

**Files:**
- Test: `tests/test_docs_consistency.py`

**Step 1: Run targeted verification**

Run: `python -m pytest -q tests/test_docs_consistency.py`

Expected: PASS

**Step 2: Review git diff**

Run: `git diff -- README.md CHANGELOG.md .github docs/public-release-package.md SUPPORT.md`

Expected: Only public release package files changed in this pass
