"""A descriptor reader must not block atomic publication of a new owner."""

from concurrent.futures import ThreadPoolExecutor
import io
import os
import sys
import threading

import pytest

from mnemos import _descriptor_io, hook_client, service


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing semantics")
@pytest.mark.parametrize("consumer", ["service", "hook"])
@pytest.mark.parametrize("filename", ["runtime.json", "runtime-\u00e9-\U0001f680.json"])
def test_open_descriptor_reader_allows_atomic_owner_replacement(
    tmp_path, monkeypatch, consumer, filename
):
    path = tmp_path / filename
    old = {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": 12345,
        "pid": 1,
        "token": "old-owner",
        "hook_protocol": 1,
        "adapter_host": "codex",
        "environment_sha256": hook_client.environment_fingerprint(),
    }
    new = {**old, "token": "new-owner"}
    service._write_private_json(path, old)
    identity = path.stat()
    real_open = io.open
    reading = threading.Event()
    release = threading.Event()
    held = []

    class DelayedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, *args, **kwargs):
            reading.set()
            assert release.wait(5), "Test did not release descriptor read"
            return self.stream.read(*args, **kwargs)

    def open_observed(*args, **kwargs):
        stream = real_open(*args, **kwargs)
        if stream.readable() and not reading.is_set():
            info = os.fstat(stream.fileno())
            if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
                assert not os.get_inheritable(stream.fileno())
                held.append(stream)
                return DelayedReader(stream)
        return stream

    monkeypatch.setattr(io, "open", open_observed)
    tokens = []

    def post(_port, token, _wire, _timeout):
        tokens.append(token)
        return None  # No request was sent; this check only reads the descriptor.

    monkeypatch.setattr(hook_client, "_post", post)

    def read():
        if consumer == "service":
            return service.read_runtime_descriptor(path)
        assert (
            hook_client.request_hook("codex", "Stop", {}, cwd=str(tmp_path), descriptor_path=path)
            is None
        )
        return {"token": tokens[-1]}

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read)
        try:
            assert reading.wait(3)
            service._write_private_json(path, new, replace_timeout=0)
        finally:
            release.set()
        assert future.result(timeout=3)["token"] == "old-owner"
    assert held and all(stream.closed for stream in held)
    assert read()["token"] == "new-owner"
    assert not list(tmp_path.glob(".*.tmp"))


def test_descriptor_text_preserves_utf8_and_reports_missing_file(tmp_path):
    path = tmp_path / "runtime.json"
    content = '{"value": "\u00e9-\U0001f680"}\n'
    path.write_text(content, encoding="utf-8")
    assert _descriptor_io.read_descriptor_text(path) == content
    path.unlink()
    with pytest.raises(FileNotFoundError):
        _descriptor_io.read_descriptor_text(path)


def test_invalid_utf8_reader_closes_before_next_publication(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        _descriptor_io.read_descriptor_text(path)
    replacement = tmp_path / "new.json"
    replacement.write_text("new")
    # The legacy operation fails on Windows if even a delete-sharing reader
    # remains open. Its success here checks cleanup after a decoding failure.
    os.replace(replacement, path)
    assert path.read_text() == "new"


def assert_native_handle_closed(handle):
    import ctypes

    query = ctypes.WinDLL("kernel32", use_last_error=True).GetHandleInformation
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    query.restype = ctypes.c_int
    flags = ctypes.c_ulong()
    assert not query(handle, ctypes.byref(flags))
    assert ctypes.get_last_error() == 6  # ERROR_INVALID_HANDLE


@pytest.mark.skipif(sys.platform != "win32", reason="Windows handle ownership")
@pytest.mark.parametrize("stage", ["crt", "stream"])
def test_reader_closes_current_owner_if_transfer_fails(tmp_path, monkeypatch, stage):
    import msvcrt

    path = tmp_path / "runtime.json"
    path.write_text("old")
    handles = []
    descriptors = []

    def fail_transfer(value, *args, **kwargs):
        if stage == "crt":
            handles.append(value)
        else:
            descriptors.append(value)
            handles.append(msvcrt.get_osfhandle(value))
        raise RuntimeError("injected ownership transfer failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            msvcrt if stage == "crt" else os,
            "open_osfhandle" if stage == "crt" else "fdopen",
            fail_transfer,
        )
        with pytest.raises(RuntimeError, match="injected ownership transfer failure"):
            _descriptor_io.read_descriptor_text(path)
    assert len(handles) == 1
    assert_native_handle_closed(handles[0])
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows unsupported rename codes")
@pytest.mark.parametrize("code", [1, 50, 87])
def test_unsupported_extended_rename_preserves_original_retry_error(tmp_path, monkeypatch, code):
    import ctypes

    original = ctypes.WinError(5)

    def legacy(*args):
        raise original

    def unsupported(*args):
        raise ctypes.WinError(code)

    monkeypatch.setattr(os, "replace", legacy)
    monkeypatch.setattr(_descriptor_io, "_replace_open_windows_file", unsupported)
    with pytest.raises(PermissionError) as raised:
        _descriptor_io.replace_descriptor(tmp_path / "new", tmp_path / "old")
    assert raised.value is original


@pytest.mark.skipif(sys.platform != "win32", reason="Windows handle ownership")
def test_failed_extended_rename_closes_source_and_preserves_destination(tmp_path, monkeypatch):
    import ctypes
    from types import SimpleNamespace

    source = tmp_path / "new.json"
    source.write_text("new")
    target = tmp_path / "directory"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    create.restype = ctypes.c_void_p
    handles = []

    def observed_create(*args):
        handle = create(*args)
        handles.append(handle)
        return handle

    proxy = SimpleNamespace(
        CreateFileW=observed_create,
        CloseHandle=kernel.CloseHandle,
        SetFileInformationByHandle=kernel.SetFileInformationByHandle,
    )
    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: proxy)
        with pytest.raises(OSError):
            _descriptor_io._replace_open_windows_file(source, target)
    assert len(handles) == 1
    assert_native_handle_closed(handles[0])
    assert source.read_text() == "new"
    assert sentinel.read_text() == "keep"
