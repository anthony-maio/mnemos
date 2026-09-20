from pathlib import Path
import hashlib,json,shutil
repo=Path(__file__).resolve().parent.parent
source=Path("C:/Users/Ant/AppData/Local/Temp/mnemos-source-association-measurement-v1")
destination=repo/"docs/validation/2026-09-20/source-association-measurement-v1"
assert not destination.exists()
destination.mkdir(parents=True)
for path in source.rglob("*"):
 if path.is_file():
  target=destination/path.relative_to(source)
  target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(path,target)
helpers=destination/"helpers";helpers.mkdir()
for path in [Path(__file__).resolve(),repo/".venv/verify_source_association_measurement_v1.py"]:
 shutil.copyfile(path,helpers/path.name)
for name in ["source-association-fixture-root-v1.xml","source-association-view-measurement-root-v3.xml","source-association-lifecycle-root-v1.xml"]:
 shutil.copyfile(repo/".venv"/name,destination/name)
(destination/".gitattributes").write_text("** -text -whitespace\n",encoding="ascii")
files={p.relative_to(destination).as_posix():{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(destination.rglob("*")) if p.is_file() and p.name!=".gitattributes"}
(destination/"index.json").write_text(json.dumps({"source_commit":"f163db1aa155b3cfa5e0d6bd2e8abedc52b2240d","files":files},indent=2)+"\n",encoding="utf-8")
print(json.dumps({"path":str(destination),"indexed_files":len(files),"bytes":sum(item["bytes"] for item in files.values())}))
