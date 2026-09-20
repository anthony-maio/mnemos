from pathlib import Path
import collections
import hashlib
import itertools
import json
import math
import sys
import zipfile

root=Path(sys.argv[1]).resolve()
archive=zipfile.ZipFile(root) if root.is_file() else None
read=lambda name: archive.read(name) if archive is not None else (root/name).read_bytes()
report=json.loads(read("measurement.json"))
assert report["measurement_status"]=="complete" and report["execution_complete"] is True
assert report["measurement_inputs_match_head_before"] is True
assert report["measurement_inputs_match_head_after"] is True
manifest=Path(__file__).resolve().parent.parent/"benchmarks/local_memory/datasets/source-association-development-v1.json"
if not manifest.exists():
    manifest=Path(sys.argv[2]).resolve()
case_data=json.loads(manifest.read_text())
cases={item["id"]:item for item in case_data["graph_cases"]}
arms=("semantic","source_association","combined")
rows=report["rows"]
assert len(rows)==45
assert {(r["case_id"],r["arm"],r["repetition"]) for r in rows}==set(itertools.product(cases,arms,range(3)))
for name,expected in report["artifacts"].items():
    data=read(name)
    assert len(data)==expected["bytes"] and hashlib.sha256(data).hexdigest()==expected["sha256"],name
failures=collections.Counter()
for row in rows:
    case=cases[row["case_id"]]
    allowed={fact["id"] for fact in case["facts"]}
    pool=row["candidate_pool"]
    assert set(pool["union_ids"])==set(pool["vector_ids"])|set(pool["lexical_ids"])
    assert set(pool["union_ids"])<=allowed
    assert row["target"]["direct_candidate"]==(case["target_id"] in pool["union_ids"])
    assert row["target"]["in_top_k"]==(case["target_id"] in row["returned_ids"])
    for score in row["candidate_scores"]:
        assert score["id"] in allowed
        assert math.isclose(score["graph_contribution"],score["graph_energy"]*0.1,rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(score["total"],score["semantic"]+score["graph_contribution"],rel_tol=1e-12,abs_tol=1e-12)
    for edge in row["typed_provenance"]:
        active=edge["components"]
        assert active
        if row["arm"]!="combined": assert active==[row["arm"]]
        assert edge["effective_weight"]==max(edge[name] for name in active)
        left,right=edge["pair"]
        assert row["edges"][left][right]==edge["effective_weight"]
        if "source_association" in active:
            assert edge["source_relations"]
            for relation in edge["source_relations"]:
                assert set(relation["fact_ids"])==set(edge["pair"])
                assert relation["source_event_id"]
                assert len(relation["evidence_ids"])>=2
    controls=row["control_observations"]
    assert controls["edge_cut"]["candidate_pool"]==pool
    for label,value in row["controls"].items():
        if value is False: failures[label]+=1
    if "lifecycle" in controls:
        lifecycle=controls["lifecycle"]
        assert lifecycle["controls"]["source_preserved_pass"] is False
        final=lifecycle["steps"]["restart"]
        actions=case["lifecycle_actions"]
        forgotten=actions[1]["fact_id"]
        assert next(f for f in final["facts"] if f["id"]==forgotten)["present"] is False
        source=actions[1]["source_event_id"]
        assert source not in {event["id"] for event in final["episodic_events"]}
        assert source in final["suppressions"]["source_event_ids"]
        corrected=next(f for f in final["facts"] if f["id"]==actions[0]["fact_id"])
        assert corrected["version"]==2 and corrected["content"]==actions[0]["new_content"]
assert report["controls_pass"]==(not failures)
print(json.dumps({"status":"verified","source_commit":report["source_commit"],"rows":len(rows),"independent_cases":len(cases),"coverage_pass":report["coverage_pass"],"control_failures":dict(failures),"summary_by_arm":report["summary_by_arm"],"artifact_files":len(report["artifacts"]),"measurement_sha256":hashlib.sha256(read("measurement.json")).hexdigest()},sort_keys=True))
