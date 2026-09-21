import sys, sqlite3, json
from pathlib import Path
sys.path.insert(0, "/tmp/mnemos-hermes-compaction-origin-candidate-v1")
from hermes_state import SessionDB
p = Path("/tmp/mnemos-hermes-growth-measurement-v1/measure.sqlite")
if p.exists(): p.unlink()
db = SessionDB(p)
db.create_session("growth", "cli", cwd="/tmp/growth")
msgs = [{"role":"user","content":"fixed user turn"},{"role":"assistant","content":"fixed assistant turn"},{"role":"tool","content":"tool output","tool_name":"x"}]
db.append_messages_batch("growth", msgs)
def counts(label):
    with sqlite3.connect(str(p)) as c:
        vals = {}
        for t in ("messages","message_origins","message_evidence_components"):
            vals[t] = c.execute("select count(*) from " + t).fetchone()[0]
        vals["active"] = c.execute("select count(*) from messages where active=1").fetchone()[0]
        vals["compacted"] = c.execute("select count(*) from messages where compacted=1").fetchone()[0]
        vals["rewind"] = c.execute("select count(*) from messages where active=0 and compacted=0").fetchone()[0]
        vals["page_count"] = c.execute("pragma page_count").fetchone()[0]
        vals["page_size"] = c.execute("pragma page_size").fetchone()[0]
        vals["db_bytes"] = vals["page_count"] * vals["page_size"]
        print(label, json.dumps(vals, sort_keys=True))
for i in range(1, 21):
    compact = [{"role":"user","content":"[summary generation %d]" % i,"_compressed_summary":True}]
    db.archive_and_compact("growth", compact, tail_count=1)
    counts("compact-%02d" % i)
db.close()
db = SessionDB(p)
print("reopen_messages", len(db.get_messages("growth", include_compacted=True)))
print("reopen_evidence", len(db.get_memory_evidence("growth")))
row = db.get_messages("growth", include_compacted=False)[-1]
rid = int(row["id"])
print("edit_row", rid, db.set_user_message_content("growth", rid, "edited fixed user turn"))
db.archive_and_compact("growth", [{"role":"user","content":"[post-edit summary]","_compressed_summary":True}], tail_count=1)
print("post_edit_evidence", len(db.get_memory_evidence("growth")))
counts("post-edit")
db.close()
