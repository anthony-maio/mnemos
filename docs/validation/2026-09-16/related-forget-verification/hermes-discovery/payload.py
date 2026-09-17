
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import zipfile

wheel = Path("/mnt/d/Development/mnemos/mnemos/.worktrees/functional-local-memory/.venv/related-forget-checks-v1/dist/mnemos_memory-0.7.0-py3-none-any.whl")
expected_project_name = 'mnemos-memory'
expected_dist_info_prefix = 'mnemos_memory'
expected_wheel_sha256 = '7d723e1bbcd095dfb9eb06f1490e10ef2699db42a55263f95df6d4b3dfe27c12'
expected_source_adapter_sha256 = '1a15f936393c3905322455c513c1cf09d6c2dc691c59422adf7866ba9d738eb2'
expected_source_hashes = {'mnemos/__init__.py': 'c33a887cd18ff275c73f0b2f13e3b60ca6f9c1b3c62e6f05ae3165c0ce85670e', 'mnemos/__main__.py': '67cda77fe285e9b33d67c36f521701cee9847dec705074249265ec5e2df69eaf', 'mnemos/adapters/claude_code.py': 'd0dfe500af60387d7486cba2604596bc14983349ed1bb80461168ebc26a4722e', 'mnemos/adapters/codex.py': '56b9bb227b9131b0f19b4d41b074c04d992d7b9a74bd6f08895610db6453d998', 'mnemos/adapters/diagnostics.py': '2fde394c9a9f47faee74181219bbbdfdca7158b516c0568845e0d6f64174fd19', 'mnemos/adapters/hermes.py': '1a15f936393c3905322455c513c1cf09d6c2dc691c59422adf7866ba9d738eb2', 'mnemos/antigravity.py': '76155276bbe660b317a24cfe796205018a9cdd0cd1d15e410168715cf008b698', 'mnemos/backup.py': '8d269fd553993d46af6ba8eef417994e17622ef0da345ebae0d592b5279a7fa1', 'mnemos/benchmark.py': 'b2caff2084e3dfd4c3cf911944e6098fbcb6f9da639636b1a7123c9e42440091', 'mnemos/bridge.py': '4d79d9734bade1d5eebc2bc81fadcbccde6092fb2ce2f2ea6f5ae8791113f286', 'mnemos/claude_identity.py': '0336e7561c6680d9d6a9aeaabdc6ac3127f994dbe5bb4adf7441a17639eb3e7a', 'mnemos/cli.py': '1a9d85f37593bcd4460ed986a8bdcc177b3842d82e5e9409e6fc22325e2a81be', 'mnemos/client.py': '489709947f462bfbf512c18817da5fb9afd4bc8d16459032d553430b3acec95b', 'mnemos/config.py': '02c48310a7e51e01b79245d89560071bf2de6c35089f4dd19ed723a402cf3912', 'mnemos/contracts.py': 'af0b4a97ea53cf33c67a21b36f9276d7c4f8001db0944a8f2500baaec6edb99e', 'mnemos/control_plane.py': '1bbce804c6875cb0a3c4988db11741130fce9ff6f5111f0b3c4116398ababa16', 'mnemos/curation.py': '114e4dbdd0ad89d97b1f94559d1ec7ed1aa5b1c8e9f991763f51d8240eaa5b41', 'mnemos/engine.py': '8c245f503278089827ff301e3975d9b40ddd9fa33a8568c7a09ed0c8ff703901', 'mnemos/health.py': 'aa03d3dd4e97de4522476e14fd8dbdbcce12ed6027cdf79e150f37665946d8fa', 'mnemos/hook_autostore.py': '97c72c1a05de7b9422f44c866e97e31599c47c5b563cdb77986a3fa3ceee2079', 'mnemos/hosts.py': 'f2ceffd68f0351b92c149f62aa48f3c7600a8343097b8126f4aacf4bb822e802', 'mnemos/inspectability.py': '2c654e4a6351e6ae94fcaddbd5ca58a13bc92938b8dd36d125ab4dc8253e487c', 'mnemos/jobs.py': '05fdb84eeab86e7ddca061ab4d7c5a81ccd258c481f5904bcdee0c7df875549a', 'mnemos/maintenance.py': 'ffd97b7f405422b775f161bb3b829ba9ce360758a1e5a5aef7a11be0d0fdb0e4', 'mnemos/managed_cli.py': 'cf76b08bb0fd2d12a5d9ab6e16aa6f65220c6f52f0120cf7aa8117283c6040c7', 'mnemos/mcp_server.py': '8b54e6a45a98cd09248f43beae0b47feb92e492f672fbda7436005c696eb0754', 'mnemos/memory_safety.py': '112d8439c7aef8c014b57069ff2b4450afb447d5143066f555abc9fccab6599e', 'mnemos/memory_service.py': 'd30c85ebe9246244c67a1fee479b94c9c326613546012cd5f5a2b0d023911195', 'mnemos/modules/__init__.py': '50895ef983cd017a898783f081d0bdbf26de050d80ccff056d462e47e06f1891', 'mnemos/modules/affective.py': '9a4a2093915fe350b646aa9a64cc5b44a31897f8fbbea437bfe22772317608cb', 'mnemos/modules/mutable_rag.py': '141b74c225922c6ea935ab8c459e1863f8c55baa74032f0098652a5f8d4049f7', 'mnemos/modules/sleep.py': '8d6bcaf30d2907fb1f9e7069e1e147ee1cfd4a90755284f562b29807c783f4a6', 'mnemos/modules/spreading.py': 'e16dc4160107898cea4c26ae5655f3d0995be66fdccdac759c985ec7237fbf97', 'mnemos/modules/surprisal.py': '44103fd642097ee4c8e15b76696e6ff1ea7203f8665a92adcc4a43a894d49c4a', 'mnemos/observability.py': '486882aa833d7d65040ad2cdccf32789e1ddb477ea09ad9dbf9519fa8be59fcc', 'mnemos/py.typed': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'mnemos/reindex.py': '9567e8d70b6129ab4844500f928447c7a9499f7971caaff79a4fa16a6145d12a', 'mnemos/retrieval.py': '6a317cd0bccde1dd466b589e544d4e82979cab4334e64bc3d378101e1fc4dbc0', 'mnemos/routing_review.py': '9426ab700e800df87d66f43e85e88224de5133487af616fc9d4be083d531d1d3', 'mnemos/runtime.py': 'fcae65b8022728af3d6e2d3cdd0334650ea955b05eec3a06908374206b07a01e', 'mnemos/scopes.py': 'dec25405fead5441f7279d8cbc9f379539f74f4df1f8110d06a23be7cb3c2b60', 'mnemos/service.py': '5d295b8c3ad9b3fd1e9fae4f5ebba9f2b74ddf72dd745c8228222968fbb96450', 'mnemos/settings.py': '742e077e787ff3e2b0fefcb1664fcb89898c9d78daf0f5340694b85ffaa3bcb3', 'mnemos/types.py': '56c54698412d4e2072b736235b2fc629282bd3adb142053a0691cd887d7a9293', 'mnemos/ui/app.js': '2804164513fad0d5bcec4cad6f29257487a5d04289d27e95da5d2a3c75744c50', 'mnemos/ui/index.html': 'c5d7fd846d9f5d0bb2b02ba02208b4534c20822fa2ff3fb76a6f73e5a478556e', 'mnemos/ui/styles.css': '9b7e877b0b2e8752dce1811ede2b860c5dc7a9bbb9a0224ab024afe3e40d85d1', 'mnemos/ui_server.py': '20dc3b2766a240e277c64d501b87895ae06aafff8cb352ea217ce622c4bb78e5', 'mnemos/utils/__init__.py': 'c22e6e1d78c80c25f0b2823f0ca0eb7eeec5deb05834d3940fff79c73a3b0483', 'mnemos/utils/embeddings.py': '8726ef2761973e54ae2f932086de6b25eb8a847a33fcc70ccb8e7357baa46307', 'mnemos/utils/llm.py': '42fc7901ececdbbcaf7b7397ff78005c577d2263e11dbf8145619218e33070bf', 'mnemos/utils/reliability.py': 'f6faa3cf5198fa5213e675a907dce4b455b5f996941168106e88e14753e2f106', 'mnemos/utils/storage.py': '6b8babe5843731029d8641ab09fb92654202f436b797f963620af36e98c820b2'}
expected_asset_hashes = {'mnemos/resources/agents/memory-orchestrator.md': 'b6a8c980a99f05ed2f95afae59f7b9d1e0d4b4c83737963e3bf6fcb61b27e2b1', 'mnemos/resources/agents/mnemos-curator.md': '71721de98145cf443430cfc08d990d77e6eee667529b384a8c336b94b725b7fd', 'mnemos/resources/agents/mnemos-recall.md': '10bb740e1758e7dca3f2cefeeeb05371ad243dbf963c53ee6c061387317a5fed'}
hermes_source = Path("/home/anthony/.hermes/hermes-agent")
sys.dont_write_bytecode = True

def normalized_sha256(path):
    raw = path.read_bytes()
    try:
        data = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except UnicodeDecodeError:
        data = raw
    return hashlib.sha256(data).hexdigest()

actual_wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
if actual_wheel_sha256 != expected_wheel_sha256:
    raise AssertionError("wheel hash does not match the requested exact hash")

with tempfile.TemporaryDirectory(prefix="mnemos-hermes-wheel-") as raw:
    root = Path(raw)
    home = root / "hermes-home"
    project = root / "project"
    extracted = root / "wheel"
    registry = root / "registry.json"
    home.mkdir()
    project.mkdir()
    (home / "plugins").mkdir()
    registry.write_text(
        '{"schema_version": 1, "projects": {}, "workspaces": {}, "profiles": ["probe-user"]}',
        encoding="utf-8",
    )
    (home / "config.yaml").write_text("memory:\n  provider: mnemos\n", encoding="utf-8")
    extracted.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)
    dist_infos = [
        path for path in extracted.glob("*.dist-info")
        if path.name.startswith(expected_dist_info_prefix + "-")
    ]
    if len(dist_infos) != 1:
        raise AssertionError("wheel must contain exactly one project dist-info directory")

    package_files = sorted(
        path.relative_to(extracted).as_posix()
        for path in extracted.glob("mnemos/**/*")
        if path.is_file()
    )
    expected_source_files = sorted(expected_source_hashes)
    expected_asset_files = sorted(expected_asset_hashes)
    expected_files = sorted(expected_source_files + expected_asset_files)
    if package_files != expected_files:
        raise AssertionError("wheel package file set differs from the clean source snapshot")
    with zipfile.ZipFile(wheel) as archive:
        wheel_package_members = sorted(
            item.filename
            for item in archive.infolist()
            if not item.is_dir() and item.filename.startswith("mnemos/")
        )
        if wheel_package_members != package_files:
            raise AssertionError("extracted mnemos package files differ from wheel members")
        for relative in package_files:
            if (extracted / relative).read_bytes() != archive.read(relative):
                raise AssertionError(f"extracted wheel bytes differ for {relative}")
    for relative in expected_source_files:
        extracted_path = extracted / relative
        if normalized_sha256(extracted_path) != expected_source_hashes[relative]:
            raise AssertionError(f"wheel package file differs from clean source: {relative}")
    for relative in expected_asset_files:
        extracted_path = extracted / relative
        if hashlib.sha256(extracted_path.read_bytes()).hexdigest() != expected_asset_hashes[relative]:
            raise AssertionError(f"packaged asset differs from clean source: {relative}")

    # env -i is supplied by the parent command. These are the only runtime
    # variables introduced for the disposable provider check.
    os.environ.clear()
    os.environ.update({
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(root / "home"),
        "TMPDIR": str(root),
        "HERMES_HOME": str(home),
        "HERMES_ENABLE_PROJECT_PLUGINS": "0",
        "MNEMOS_SCOPE_REGISTRY": str(registry),
        "MNEMOS_USER_PROFILE_ID": "probe-user",
    })
    os.chdir(project)
    sys.path[:] = [item for item in sys.path if item not in ("", str(extracted), str(hermes_source))]
    sys.path.insert(0, str(hermes_source))
    sys.path.insert(0, str(extracted))

    network_calls = []

    def blocked_network(*args, **kwargs):
        network_calls.append("socket.connect")
        raise AssertionError("wheel verification attempted socket.connect")

    socket.socket.connect = blocked_network
    import mnemos.client as mnemos_client

    service_calls = []

    def blocked_service_method(*args, **kwargs):
        service_calls.append("MnemosClient")
        raise AssertionError("wheel verification attempted a MnemosClient service call")

    for method_name in ("request", "replay_outbox", "capture", "retrieve", "close"):
        setattr(mnemos_client.MnemosClient, method_name, blocked_service_method)

    from plugins.memory import find_provider_dir, find_provider_entry_point, load_memory_provider
    import mnemos.adapters.hermes as mnemos_hermes

    entry_point = find_provider_entry_point("mnemos")
    if entry_point is None:
        raise AssertionError("wheel did not expose mnemos memory-provider entry point")
    if entry_point.value != "mnemos.adapters.hermes:register":
        raise AssertionError(f"unexpected memory-provider entry point: {entry_point.value!r}")
    if not callable(entry_point.load()):
        raise AssertionError("memory-provider entry point did not load a callable")
    provider_dir = find_provider_dir("mnemos")
    # Hermes returns a filesystem directory only for package entry points whose
    # module origin is __init__.py. Mnemos exposes a module entry point, so the
    # documented loader result is None; the distribution and imported module
    # origins below provide the wheel binding proof.
    if provider_dir is not None:
        raise AssertionError("entry-point provider unexpectedly resolved as a filesystem provider")
    if list((home / "plugins").iterdir()):
        raise AssertionError("disposable Hermes home unexpectedly contains another provider directory")
    distribution = getattr(entry_point, "dist", None)
    if distribution is None or distribution.metadata.get("Name") != expected_project_name:
        raise AssertionError("entry point is not owned by the project distribution")
    distribution_root = Path(distribution.locate_file("")).resolve()
    if distribution_root != extracted.resolve():
        raise AssertionError("entry-point distribution does not originate in the extracted wheel")
    provider = load_memory_provider("mnemos", register_skills=False)
    adapter_path = Path(mnemos_hermes.__file__).resolve()
    expected_adapter_path = (extracted / "mnemos" / "adapters" / "hermes.py").resolve()
    if adapter_path != expected_adapter_path:
        raise AssertionError(f"adapter imported outside extracted wheel: {adapter_path}")
    adapter_normalized = normalized_sha256(adapter_path)
    if adapter_normalized != expected_source_adapter_sha256:
        raise AssertionError("wheel adapter does not match the verified source snapshot")
    if provider is None or not provider.is_available():
        raise AssertionError("actual wheel entry point did not load an available provider")
    if network_calls or service_calls:
        raise AssertionError("wheel discovery contacted a network or Mnemos service")
    source_revision = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(hermes_source), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if revision.returncode == 0:
            source_revision = revision.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    print(json.dumps({
        "hermes_version": metadata.version("hermes-agent"),
        "hermes_source_revision": source_revision,
        "hermes_source_revision_status": "observed" if source_revision else "unavailable",
        "mnemos_version": metadata.version(expected_project_name),
        "project_name": expected_project_name,
        "distribution_name": distribution.metadata.get("Name"),
        "entry_point_value": entry_point.value,
        "entry_point_origin": str(adapter_path),
        "entry_point_origin_matches_extracted_wheel": adapter_path == expected_adapter_path,
        "provider_dir": None,
        "wheel_sha256": actual_wheel_sha256,
        "adapter_sha256_normalized": adapter_normalized,
        "source_files_checked": len(expected_source_files),
        "packaged_assets_checked": len(expected_asset_files),
        "provider_loaded": True,
        "provider_available": True,
        "network_calls": network_calls,
        "service_calls": service_calls,
        "bridge_env_present": "MNEMOS_BRIDGE_COMMAND" in os.environ,
        "pythonpath_env_present": "PYTHONPATH" in os.environ,
        "service_contact_attempted": False,
    }, sort_keys=True))
