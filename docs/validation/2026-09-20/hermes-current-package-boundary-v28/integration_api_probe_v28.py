from __future__ import annotations
import asyncio
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from zipfile import ZipFile

root = Path(tempfile.mkdtemp(prefix='mnemos-hermes-root-new-v28-'))
os.environ['HERMES_HOME'] = str(root / 'home')
os.environ['TMPDIR'] = os.environ['TEMP'] = os.environ['TMP'] = '/tmp'
for key in tuple(os.environ):
    if any(part in key.upper() for part in ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN')) or key.startswith('MNEMOS_'):
        os.environ.pop(key)
Path(os.environ['HERMES_HOME']).mkdir()

from agent.memory_manager import MemoryManager
from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_state import SessionDB
from run_agent import AIAgent
from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.client import MnemosClient
from mnemos.scopes import ScopeRegistry
import cli as cli_module

# UI/model selection configuration only; memory paths and methods remain real.
cli_module.CLI_CONFIG = {'model': {'default': 'fixture', 'provider': ''}, 'agent': {}}
wheel = Path('/mnt/c/Users/Ant/AppData/Local/Temp/mnemos-package-checks-generation-settings-v2/dist/mnemos_memory-0.7.0-py3-none-any.whl')
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '8d75eef166c6e7f5558dd0a3b3c3562ac66fd7e89d9e1c8d7731688ef099260a'
installed_root = Path(inspect.getfile(MnemosHermesProvider)).parents[2]
verified_modules = 0
with ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.startswith('mnemos/') and name.endswith('.py'):
            assert (installed_root / name).read_bytes() == archive.read(name), name
            verified_modules += 1

candidate = Path(__file__).resolve().parent
patch_before = subprocess.check_output(['git', 'diff', '--binary'], cwd=candidate)
extra_sources = {name: hashlib.sha256((candidate / name).read_bytes()).hexdigest()
                 for name in ['agent/memory_source_snapshot.py']}
results = []
for corrupted in (False, True):
    case = root / ('corrupt' if corrupted else 'valid')
    case.mkdir()
    db = SessionDB(case / 'state.db')
    old = 'old-corrupt' if corrupted else 'old-valid'
    db.create_session(old, 'cli', cwd=str(case), model='fixture')
    registry = ScopeRegistry()
    registry.register_profile('profile')
    registry.register_project(case, user_profile_id='profile', enabled=True)
    outbox = case / 'outbox.json'
    client = MnemosClient(auto_start=False, outbox_path=outbox, descriptor_path=case / 'descriptor.json')
    provider = MnemosHermesProvider(client=client, registry=registry, user_profile_id='profile')
    provider.initialize(old, cwd=str(case), hermes_home=str(root / 'home'))
    manager = MemoryManager()
    manager._providers = [provider]
    # Avoid the interactive constructor and its credential/tool discovery.
    # All persistence, evidence, reset, and memory methods are actual methods.
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = manager
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = old
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._memory_store = None
    agent.context_compressor = None
    agent.model = 'fixture'
    history = [{'role': 'user', 'content': 'The build artifact requirement is cedar package format.'}]
    failure = None
    if corrupted:
        assert agent._flush_messages_to_session_db(history, conversation_history=[]) is True
        assert len(db.get_memory_evidence(old)) == 1
        with sqlite3.connect(case / 'state.db') as connection:
            changed = connection.execute("UPDATE message_evidence_components SET content = ?", ('forged durable content',)).rowcount
            assert changed == 1
        try:
            agent._memory_evidence_snapshot(old)
        except ValueError as exc:
            failure = str(exc)
        else:
            raise AssertionError('Corrupted persisted evidence was accepted')
    else:
        assert db.get_messages(old) == []
    cli = CLISessionMixin.__new__(CLISessionMixin)
    cli.session_id, cli.session_start = old, datetime.now()
    cli._session_db, cli.conversation_history, cli.agent = db, history, agent
    for key, value in {
        'model': 'fixture', 'provider': '', 'base_url': '', 'api_key': '', 'requested_provider': '',
        '_explicit_api_key': '', '_explicit_base_url': '', '_explicit_model_override': False,
        '_pending_one_turn_model_restore': None, 'reasoning_config': {}, 'service_tier': None,
        'max_turns': 1, '_resumed': False, '_pending_title': None, '_status_bar_title_checked_at': 0.0,
    }.items():
        setattr(cli, key, value)
    cli._notify_session_boundary = lambda *a, **k: None
    cli._write_terminal_breadcrumb = lambda: None
    cli._console_print = lambda *a, **k: None
    CLISessionMixin.new_session(cli, silent=True)
    assert cli.session_id != old
    assert manager.flush_pending(timeout=10)
    assert provider.session_id == cli.session_id
    assert provider._context.session_id == cli.session_id
    rows = db.get_messages(old)
    assert len(rows) == 1
    arrivals = json.loads(outbox.read_text()) if outbox.exists() else []
    if corrupted:
        assert arrivals == [], arrivals
    else:
        assert len(arrivals) == 1, arrivals
        assert arrivals[0]['operation'] == 'capture'
        payload = arrivals[0]['payload']
        assert payload['source_ref'] == f"transcript:v2:{rows[0]['id']}", payload
        assert payload['context']['session_id'] == old, payload
        assert payload['content'] == history[0]['content'], payload
    results.append({'corrupted': corrupted, 'projection_failure': failure, 'old_session': old,
                    'new_session': cli.session_id, 'provider_session': provider.session_id,
                    'old_durable_rows': len(rows), 'outbox_captures': len(arrivals)})
    manager.shutdown_all()
    db.close()
    asyncio.run(client.close())
patch_after = subprocess.check_output(['git', 'diff', '--binary'], cwd=candidate)
assert patch_before == patch_after
assert extra_sources == {name: hashlib.sha256((candidate / name).read_bytes()).hexdigest()
                         for name in extra_sources}
result = {'status': 'passed', 'root': str(root), 'cases': results,
          'candidate_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=candidate, text=True).strip(),
          'candidate_patch_sha256': hashlib.sha256(patch_before).hexdigest(),
          'verified_installed_python_modules': verified_modules,
          'extra_sources': extra_sources,
          'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
          'limit': 'Actual CLI new_session method with UI scaffold, private durable DB and installed provider. Outbox only; service delivery and full interactive CLI not exercised.'}
output = candidate / 'integration_api_probe_v28.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
