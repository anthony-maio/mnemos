from pathlib import Path
import hashlib
import json
import shutil

repo=Path(__file__).resolve().parent.parent
source=Path("C:/Users/Ant/AppData/Local/Temp/mnemos-source-association-vectors-v1")
destination=repo / "docs/validation/2026-09-20/source-association-vectors-v1"
assert not destination.exists()
destination.mkdir(parents=True)
for path in source.rglob("*"):
    if path.is_file():
        target=destination / path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
helpers=destination / "helpers"
helpers.mkdir()
for path in (Path(__file__).resolve(),repo / ".venv/verify_source_association_vectors_v1.py",repo / "benchmarks/local_memory/prepare_source_association_vectors.py"):
    shutil.copyfile(path, helpers / path.name)
shutil.copyfile(repo / ".venv/source-association-preparation-root-v2.xml", destination / "offline-tests.xml")
(destination / ".gitattributes").write_text("** -text -whitespace\n", encoding="ascii")
files={path.relative_to(destination).as_posix():{"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(destination.rglob("*")) if path.is_file() and path.name!=".gitattributes"}
(destination / "index.json").write_text(json.dumps({"schema_version":"source-association-vector-archive-v1","source_commit":"326322f02fc372adaced52dccf8b316c95cedc04","files":files},indent=2)+"\n",encoding="utf-8")
for relative,expected in files.items():
    data=(destination/relative).read_bytes()
    assert len(data)==expected["bytes"] and hashlib.sha256(data).hexdigest()==expected["sha256"]
print(json.dumps({"path":str(destination),"indexed_files":len(files),"bytes":sum(x["bytes"] for x in files.values())}))
