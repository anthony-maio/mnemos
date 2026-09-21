from datetime import datetime, timezone
import hashlib
from candidate.source_selection import SelectionInventory, render_selections
from mnemos.types import EpisodicEvent, MemoryChunk

def materialize(case):
    events = [EpisodicEvent(
        id=e['id'], event_fingerprint=hashlib.sha256(e['content'].encode()).hexdigest(),
        role=e['role'], content=e['content'], normalized_content=e['content'].lower(),
        scope=e['scope'], scope_id=e['scope_id'], source=e['source'],
        metadata={'user_profile_id': e['profile_id']},
        occurred_at=datetime(2026, 9, 21, 12, tzinfo=timezone.utc)) for e in case['source_events']]
    candidates = render_selections(SelectionInventory.model_validate({'selections': case['source_selections']}), events)
    targets = [MemoryChunk(id=t['id'], content=t['content'], version=t['version'],
                           metadata={'user_profile_id':t['profile_id'], 'scope':t['scope'], 'scope_id':t['scope_id']})
               for t in case['target_facts']]
    return candidates, events, targets

