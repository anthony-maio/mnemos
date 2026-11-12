"""Model-free input admission for the inactive local support candidate.

The frozen computation covered at most eight owned sources per fact. This
policy does not extend that result to every combination within the limits.
It is not wired into production retrieval or any live installation.
"""

import json
import importlib.util
from pathlib import Path
import sys

_name = "local_support_v2_capacity"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _name, Path(__file__).with_name("bidirectional_support_capacity.py")
    )
    capacity = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = capacity
    _spec.loader.exec_module(capacity)
else:
    capacity = sys.modules[_name]


def frame_local_support(value, tokenizer, binding):
    """Retain the whole cohort or reject it before any model computation.

    Reuse the frozen strict schema, 8,192-token complete native frame and
    64,000-byte payload limits. Inspect the validated payload for the narrower
    source inventory. A failure is operational unavailability, not semantic
    abstention; the evaluator must keep answerable failures in denominators.
    """
    framed = capacity.frame_request(value, tokenizer, binding)
    candidates = json.loads(framed.payload_json)["candidates"]
    if any(len(candidate["sources"]) > 8 for candidate in candidates):
        raise ValueError("Source inventory exceeds the eight-source candidate envelope")
    return framed
