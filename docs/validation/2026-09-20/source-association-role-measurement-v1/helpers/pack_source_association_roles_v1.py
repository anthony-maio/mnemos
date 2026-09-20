from pathlib import Path
import hashlib
import json
import shutil
import zipfile

repo = Path(__file__).resolve().parent.parent
source = Path('C:/Users/Ant/AppData/Local/Temp/mnemos-source-association-role-measurement-v1')
destination = repo / 'docs/validation/2026-09-20/source-association-role-measurement-v1'
assert not destination.exists()
destination.mkdir(parents=True)
(destination / 'helpers').mkdir()
(destination / '.gitattributes').write_text('** -text -whitespace\n', encoding='ascii')
members = {}
zip_path = destination / 'raw-evidence.zip'
with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(source.rglob('*')):
        if path.is_file():
            raw = path.read_bytes()
            name = path.relative_to(source).as_posix()
            archive.writestr(name, raw)
            members[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
with zipfile.ZipFile(zip_path) as archive:
    assert set(archive.namelist()) == set(members)
    for name, expected in members.items():
        raw = archive.read(name)
        assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
report = json.loads((source / 'measurement.json').read_bytes())
summary = {key: value for key, value in report.items() if key not in {'rows', 'artifacts'}}
summary['control_failures'] = [
    {key: row[key] for key in ('embedding_contract', 'case_id', 'arm', 'repetition')}
    | {'failed_controls': [key for key, value in row['controls'].items() if value is False]}
    for row in report['rows'] if any(value is False for value in row['controls'].values())
]
summary['measurement_sha256'] = hashlib.sha256((source / 'measurement.json').read_bytes()).hexdigest()
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
for name in ('pack_source_association_roles_v1.py', 'verify_source_association_roles_v1.py'):
    shutil.copyfile(repo / '.venv' / name, destination / 'helpers' / name)
for name in ('source-association-role-verification.json', 'source-association-role-offline.xml',
             'source-association-role-black.log', 'source-association-role-mypy.log',
             'role-runner-red.xml', 'role-runner-green.xml'):
    shutil.copyfile(repo / '.venv' / name, destination / name)
files = {path.relative_to(destination).as_posix(): {
    'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
} for path in sorted(destination.rglob('*')) if path.is_file() and path.name != '.gitattributes'}
(destination / 'index.json').write_text(json.dumps({
    'source_commit': report['source_commit'], 'files': files, 'archive_members': members
}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'archive_members': len(members), 'indexed_files': len(files),
                  'zip_bytes': zip_path.stat().st_size,
                  'zip_sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest()}))
