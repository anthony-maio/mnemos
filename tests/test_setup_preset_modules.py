"""Exercise the actual setup form's module payload in a small DOM fixture."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from mnemos.control_plane import SETUP_PRESETS


def test_starter_preset_explicitly_disables_every_experiment():
    assert SETUP_PRESETS[0]["payload"]["modules"] == {
        "surprisal": False,
        "mutable_rag": False,
        "affective": False,
        "sleep": False,
        "spreading": False,
    }


def test_form_preserves_custom_modules_until_preset_is_selected(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the setup form")
    source = Path(__file__).parents[1] / "mnemos" / "ui" / "app.js"
    script = tmp_path / "preset.cjs"
    script.write_text(
        r"""
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value: "", checked: false, textContent: "", innerHTML: ""});
    return elements.get(id);
  },
  querySelectorAll() { return []; },
};
const context = vm.createContext({document, console, assert});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8").split("\ninit().catch(")[0], context);
vm.runInContext(`
  const modules = {surprisal: true, mutable_rag: false, affective: false, sleep: true, spreading: true};
  const settings = {modules, llm: {provider: "ollama"}, providers: {ollama: {}}, sleep: {}};
  const presets = ${process.argv[3]};
  fillForm({settings, presets});
  // Ordinary edits must not pin inherited global flags into a project override.
  assert.equal(collectPayload().modules, undefined);
  assert.equal(activePresetId(settings), null);
  applyPreset("starter-sqlite");
  assert.equal(JSON.stringify(collectPayload().modules), JSON.stringify(presets[0].payload.modules));
  assert.equal(activePresetId(collectPayload()), "starter-sqlite");
  assert.equal(settings.modules.spreading, true);
  applyPreset("stable-memory");
  assert.equal(collectPayload().modules.sleep, true);
  assert.equal(activePresetId(collectPayload()), "stable-memory");
`, context);
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(script), str(source), json.dumps(SETUP_PRESETS)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
