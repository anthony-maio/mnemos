"""Hook startup must not initialize providers that only the service uses."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import mnemos


@pytest.mark.parametrize("module", ["mnemos", "mnemos.client", "mnemos.adapters.codex"])
def test_capture_import_does_not_load_inference_runtime(module: str, tmp_path: Path) -> None:
    code = """
import importlib,importlib.abc,sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in {'numpy','mnemos.engine','mnemos.utils.embeddings','mnemos.utils.storage'}:
            raise AssertionError('Capture imported inference runtime: '+fullname)
sys.meta_path.insert(0,BlockRuntime())
importlib.import_module(sys.argv[1])
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(mnemos.__file__).resolve().parent.parent)
    completed = subprocess.run(
        [sys.executable, "-c", code, module],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_exports_keep_class_identity_and_star_import_contract() -> None:
    from mnemos import MnemosEngine, MnemosConfig, SQLiteStore, SimpleEmbeddingProvider
    from mnemos.config import MnemosConfig as Config
    from mnemos.engine import MnemosEngine as Engine
    from mnemos.utils.embeddings import SimpleEmbeddingProvider as Embeddings
    from mnemos.utils.storage import SQLiteStore as Store

    assert MnemosEngine is Engine
    assert MnemosConfig is Config
    assert SQLiteStore is Store
    assert SimpleEmbeddingProvider is Embeddings
    assert set(mnemos.__all__) <= set(dir(mnemos))
    exported: dict = {}
    exec("from mnemos import *", exported)
    for name in mnemos.__all__:
        assert exported[name] is getattr(mnemos, name)
    with pytest.raises(AttributeError, match="no_such_public_export"):
        getattr(mnemos, "no_such_public_export")
