"""Exclusive, flush-and-fsync evidence journal for serial development runs."""

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f'Unsupported evidence type: {type(value).__name__}')


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, default=_default).encode('utf-8')


class EvaluationJournal:
    """Never overwrites or resumes a prior run; owner supplies a private directory.

    A successful emit means file flush and fsync returned. This does not claim
    power-loss durability of directory creation or hardware write caches.
    A failed append poisons the writer: retain its bytes and start no new request.
    """

    def __init__(self, path: Path):
        self._file = path.open('xb')
        self._sequence = 0
        self._previous = '0' * 64
        self._failed = False

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        if self._failed or self._file.closed:
            raise RuntimeError('Journal is failed or closed')
        try:
            if not isinstance(kind, str) or not kind.strip() or not isinstance(data, dict):
                raise ValueError('Expected event kind and data object')
            body = dict(sequence=self._sequence, previous_sha256=self._previous, kind=kind,
                        recorded_at=datetime.now(timezone.utc).isoformat(), data=data)
            digest = hashlib.sha256(_encode(body)).hexdigest()
            self._file.write(_encode(dict(**body, sha256=digest)) + b'\n')
            self._file.flush()
            os.fsync(self._file.fileno())
            self._sequence += 1
            self._previous = digest
        except BaseException:
            self._failed = True
            raise

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_journal(path: Path) -> list[dict[str, Any]]:
    """Verify complete ordered records; never repair or discard a partial tail.

    This is integrity checking, not authenticity or proof a run completed.
    Completion requires an explicit summary plus the owner's cleanup evidence.
    """
    records = []
    previous = '0' * 64
    with path.open('rb') as stream:
        for sequence, line in enumerate(stream):
            if not line.endswith(b'\n'):
                raise ValueError('Incomplete journal tail')
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('Invalid journal record')
            digest = record.pop('sha256', None)
            if (record.get('sequence') != sequence or record.get('previous_sha256') != previous
                    or hashlib.sha256(_encode(record)).hexdigest() != digest):
                raise ValueError('Journal sequence or digest mismatch')
            records.append(dict(**record, sha256=digest))
            previous = digest
    return records
