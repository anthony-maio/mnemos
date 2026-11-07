"""Read descriptor snapshots without preventing atomic owner replacement."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TextIO


def read_descriptor_text(path: Path) -> str:
    if sys.platform != "win32":
        return path.read_text(encoding="utf-8")

    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    # GENERIC_READ; FILE_SHARE_READ | WRITE | DELETE; OPEN_EXISTING.
    # Delete sharing permits rename/replacement while this handle continues
    # reading its original file. It does not grant access or change the ACL.
    handle = create_file(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_NOINHERIT)
    except BaseException:
        close_handle(handle)
        raise
    # Ownership transfers from the native handle to the CRT descriptor, then
    # to the Python stream. Close the current owner if either transfer fails.
    try:
        stream: TextIO = os.fdopen(descriptor, "r", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        return stream.read()


def replace_descriptor(source: Path, target: Path) -> None:
    """Publish atomically, including over cooperating Windows readers."""
    try:
        os.replace(source, target)
    except OSError as original:
        if sys.platform != "win32" or getattr(original, "winerror", None) not in {5, 32, 33}:
            raise
        try:
            _replace_open_windows_file(source, target)
        except OSError as native:
            # Older Windows versions and filesystems may not support extended
            # rename flags. Retain the original bounded retry path in that case.
            if getattr(native, "winerror", None) in {1, 50, 87}:
                raise original from native
            raise


def _replace_open_windows_file(source: Path, target: Path) -> None:
    if sys.platform != "win32":
        raise OSError("Extended descriptor replacement requires Windows")

    import ctypes

    class RenameInfo(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_ulong),
            ("root_directory", ctypes.c_void_p),
            ("name_length", ctypes.c_ulong),
            ("name", ctypes.c_wchar * 1),
        ]

    name = str(target.absolute()).encode("utf-16-le", errors="surrogatepass")
    buffer = ctypes.create_string_buffer(ctypes.sizeof(RenameInfo) + len(name) + 2)
    info = RenameInfo.from_buffer(buffer)
    # REPLACE_IF_EXISTS | POSIX_SEMANTICS. Existing delete-sharing readers
    # retain the old file; later opens see the new file. ACLs still apply.
    info.flags = 0x3
    info.name_length = len(name)
    ctypes.memmove(ctypes.addressof(buffer) + RenameInfo.name.offset, name, len(name))

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    rename = kernel32.SetFileInformationByHandle
    rename.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    rename.restype = ctypes.c_int
    # DELETE; share read/write/delete; OPEN_EXISTING. The temporary file has
    # already received its private ACL before this handle is opened.
    handle = create_file(str(source), 0x10000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # FileRenameInfoEx = 22, with the native structure's alignment.
        if not rename(handle, 22, ctypes.byref(buffer), ctypes.sizeof(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close_handle(handle)
