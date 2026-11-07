"""Synchronous, bounded WinHTTP transport for the loopback hook endpoint.

Unavailable is raised only before sending. After sending, a failure is an
unknown outcome; callers must not resend through another transport.
"""

from __future__ import annotations

import ctypes
import math
from typing import Any

_DWORD = ctypes.c_uint32
_HANDLE = ctypes.c_void_p
_BOOL = ctypes.c_int32
_WORD = ctypes.c_uint16
_STRING = ctypes.c_wchar_p
_POINTER = ctypes.c_void_p
_DWORD_POINTER = ctypes.POINTER(_DWORD)


class WinHTTPUnavailable(RuntimeError):
    """The native client could not be prepared; no request bytes were sent."""


def _load_api() -> Any:
    # Load only the Windows system copy, never a DLL beside a project file.
    api = getattr(ctypes, "WinDLL")("winhttp.dll", use_last_error=True, winmode=0x00000800)
    signatures = {
        "WinHttpOpen": (_HANDLE, [_STRING, _DWORD, _STRING, _STRING, _DWORD]),
        "WinHttpSetTimeouts": (
            _BOOL,
            [_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        ),
        "WinHttpSetOption": (_BOOL, [_HANDLE, _DWORD, _POINTER, _DWORD]),
        "WinHttpConnect": (_HANDLE, [_HANDLE, _STRING, _WORD, _DWORD]),
        "WinHttpOpenRequest": (
            _HANDLE,
            [_HANDLE, _STRING, _STRING, _STRING, _STRING, _POINTER, _DWORD],
        ),
        "WinHttpSendRequest": (
            _BOOL,
            [_HANDLE, _STRING, _DWORD, _POINTER, _DWORD, _DWORD, ctypes.c_size_t],
        ),
        "WinHttpReceiveResponse": (_BOOL, [_HANDLE, _POINTER]),
        "WinHttpQueryHeaders": (
            _BOOL,
            [_HANDLE, _DWORD, _STRING, _POINTER, _DWORD_POINTER, _DWORD_POINTER],
        ),
        "WinHttpReadData": (_BOOL, [_HANDLE, _POINTER, _DWORD, _DWORD_POINTER]),
        "WinHttpCloseHandle": (_BOOL, [_HANDLE]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(api, name)
        function.restype = result
        function.argtypes = arguments
    return api


def _require(success: Any) -> None:
    if not success:
        # Numeric errors are safe to retain; server text and credentials are not.
        raise OSError(getattr(ctypes, "get_last_error")(), "WinHTTP hook transport failed")


def post(port: int, token: str, wire: bytes, *, timeout: float, limit: int) -> bytes | None:
    """Return response bytes, or None only when the initial connection failed."""
    if (
        type(port) is not int
        or not 1 <= port <= 65535
        or not token
        or len(token) > 512
        or any(not 33 <= ord(char) <= 126 for char in token)
        or not math.isfinite(timeout)
        or timeout <= 0
        or type(limit) is not int
        or not 0 < limit < 2**31
        or len(wire) > limit
    ):
        raise ValueError("Invalid loopback hook transport settings")
    api: Any = None
    handles: list[int] = []
    try:
        try:
            api = _load_api()
            session = api.WinHttpOpen("mnemos-hook/1", 1, None, None, 0)  # NO_PROXY, synchronous
            _require(session)
            handles.append(session)
            milliseconds = max(1, min(2**31 - 1, math.ceil(timeout * 1000)))
            _require(
                api.WinHttpSetTimeouts(
                    session, milliseconds, milliseconds, milliseconds, milliseconds
                )
            )
            # One connection attempt. No automatic resend after a failed connection.
            attempts = _DWORD(1)
            retries = (_DWORD * 2)(0, 0)  # WINHTTP_FAILED_CONNECTION_RETRIES
            no_pool = _BOOL(1)
            _require(
                api.WinHttpSetOption(session, 4, ctypes.byref(attempts), ctypes.sizeof(attempts))
            )
            _require(
                api.WinHttpSetOption(session, 162, ctypes.byref(retries), ctypes.sizeof(retries))
            )
            _require(
                api.WinHttpSetOption(session, 195, ctypes.byref(no_pool), ctypes.sizeof(no_pool))
            )
            connection = api.WinHttpConnect(session, "127.0.0.1", port, 0)
            _require(connection)
            handles.append(connection)
            request = api.WinHttpOpenRequest(
                connection, "POST", "/v1/dispatch", "HTTP/1.1", None, None, 0
            )
            _require(request)
            handles.append(request)
            # Disable cookies, redirects, automatic authentication and keep-alive.
            disabled = _DWORD(0xF)
            header_limit = _DWORD(65536)
            _require(
                api.WinHttpSetOption(request, 63, ctypes.byref(disabled), ctypes.sizeof(disabled))
            )
            _require(
                api.WinHttpSetOption(
                    request, 91, ctypes.byref(header_limit), ctypes.sizeof(header_limit)
                )
            )
        except (OSError, AttributeError) as exc:
            raise WinHTTPUnavailable("WinHTTP setup unavailable before dispatch") from exc

        headers = "Authorization: Bearer " + token + "\r\nContent-Type: application/json\r\n"
        body = ctypes.create_string_buffer(wire)
        sent = api.WinHttpSendRequest(request, headers, len(headers), body, len(wire), len(wire), 0)
        if (
            not sent and getattr(ctypes, "get_last_error")() == 12029
        ):  # ERROR_WINHTTP_CANNOT_CONNECT
            return None
        _require(sent)
        _require(api.WinHttpReceiveResponse(request, None))
        status, size = _DWORD(), _DWORD(ctypes.sizeof(_DWORD))
        _require(
            api.WinHttpQueryHeaders(
                request, 19 | 0x20000000, None, ctypes.byref(status), ctypes.byref(size), None
            )
        )
        if status.value != 200:
            raise ValueError("Invalid worker HTTP status")
        # WinHttpReadData can report EOF even when Content-Length was larger.
        # Our worker always sends a fixed-length body; require the whole frame.
        expected = ctypes.c_uint64()
        size = _DWORD(ctypes.sizeof(expected))
        _require(
            api.WinHttpQueryHeaders(
                request, 5 | 0x08000000, None, ctypes.byref(expected), ctypes.byref(size), None
            )
        )
        if expected.value > limit:
            raise ValueError("Worker response exceeds its limit")
        chunks: list[bytes] = []
        total = 0
        buffer = ctypes.create_string_buffer(min(65536, limit + 1))
        while True:
            count = _DWORD()
            _require(
                api.WinHttpReadData(
                    request, buffer, min(len(buffer), limit + 1 - total), ctypes.byref(count)
                )
            )
            if count.value == 0:
                if total != expected.value:
                    raise ValueError("Incomplete worker response")
                return b"".join(chunks)
            total += count.value
            if total > limit:
                raise ValueError("Worker response exceeds its limit")
            chunks.append(buffer.raw[: count.value])
    finally:
        if api is not None:
            for handle in reversed(handles):
                api.WinHttpCloseHandle(handle)
