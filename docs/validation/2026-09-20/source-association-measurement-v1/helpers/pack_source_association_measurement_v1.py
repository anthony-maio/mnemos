from pathlib import Path
import hashlib,json,shutil,zipfile
repo=Path(__file__).resolve().parent.parent
source=Path("C:/Users/Ant/AppData/Local/Temp/mnemos-source-association-measurement-v1")
destination=repo/"docs/validation/2026-09-20/source-association-measurement-v1"
zip_path=destination/"raw-evidence.zip"
assert not zip_path.exists()
members={}
with zipfile.ZipFile(zip_path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
 for path in sorted(source.rglob("*")):
  if path.is_file():
   data=path.read_bytes();name=path.relative_to(source).as_posix()
   archive.writestr(name,data)
   members[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
with zipfile.ZipFile(zip_path) as archive:
 assert set(archive.namelist())==set(members)
 for name,expected in members.items():
  data=archive.read(name)
  assert len(data)==expected["bytes"] and hashlib.sha256(data).hexdigest()==expected["sha256"]
report=json.loads((source/"measurement.json").read_text())
summary={key:report[key] for key in ["schema_version","measurement_status","execution_complete","controls_pass","coverage_pass","source_commit","started_at","finished_at","manifest_sha256","vector_artifact_sha256","protocol_sha256","model","model_digest","embedding_text_contract","options","expected_rows","independent_cases","summary_by_arm","cases_with_shared_source_relations"]}
summary["control_failures"]=[{"case_id":row["case_id"],"arm":row["arm"],"repetition":row["repetition"],"failed_controls":[key for key,value in row["controls"].items() if value is False]} for row in report["rows"] if any(value is False for value in row["controls"].values())]
summary["measurement_sha256"]=hashlib.sha256((source/"measurement.json").read_bytes()).hexdigest()
(destination/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
for path in [Path(__file__).resolve(),repo/".venv/verify_source_association_measurement_v1.py"]:
 shutil.copyfile(path,destination/"helpers"/path.name)
files={path.relative_to(destination).as_posix():{"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(destination.rglob("*")) if path.is_file() and (path.parent.name=="helpers" or path.parent==destination and path.name in {"summary.json","raw-evidence.zip","source-association-fixture-root-v1.xml","source-association-view-measurement-root-v3.xml","source-association-lifecycle-root-v1.xml"})}
(destination/"index.json").write_text(json.dumps({"source_commit":report["source_commit"],"files":files,"archive_members":members},indent=2)+"\n",encoding="utf-8")
print(json.dumps({"zip_bytes":zip_path.stat().st_size,"archive_members":len(members),"files":len(files),"zip_sha256":hashlib.sha256(zip_path.read_bytes()).hexdigest()}))
