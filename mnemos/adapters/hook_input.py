"""Read hook JSON using its wire encoding instead of the host locale."""

from __future__ import annotations

from typing import BinaryIO, TextIO, cast


def read_hook_input(stream: TextIO) -> str:
    """Decode piped hook input as UTF-8, retaining support for text streams."""
    binary = cast(BinaryIO | None, getattr(stream, "buffer", None))
    if binary is not None:
        return binary.read().decode("utf-8")
    return stream.read()
