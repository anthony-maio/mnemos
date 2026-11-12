import pytest
import json
from pathlib import Path
import subprocess
import sys


def account(cases, outcomes):
    try:
        from benchmarks.local_memory.local_support_v2_controls import account_controls
    except ModuleNotFoundError as error:
        if error.name != "benchmarks.local_memory.local_support_v2_controls":
            raise
        pytest.fail("Boundary-control accounting is not implemented")
    return account_controls(cases, outcomes)


def cases():
    return [
        dict(id="empty", supported_target_ids=[]),
        dict(id="small", supported_target_ids=["a"]),
        dict(id="oversize", supported_target_ids=["b", "c"]),
    ]


def test_answerable_unavailability_retains_case_and_target_denominators():
    # Observation order cannot change which gold targets are counted.
    result = account(
        cases(),
        [
            dict(id="oversize", availability="unavailable"),
            dict(id="empty", availability="admitted"),
            dict(id="small", availability="admitted"),
        ],
    )
    assert result["controls"] == 3
    assert result["answerable_controls"] == 2
    assert result["unavailable_answerable_controls"] == 1
    assert result["supported_target_opportunities"] == 3
    assert result["unavailable_supported_target_opportunities"] == 2
    assert result["semantic_recall"] is None
    assert result["semantic_precision"] is None


def test_missing_observation_cannot_produce_a_passing_summary():
    with pytest.raises(ValueError, match="complete"):
        account(cases(), [dict(id="empty", availability="admitted")])


def test_duplicate_observation_cannot_hide_a_missing_control():
    with pytest.raises(ValueError, match="Duplicate"):
        account(cases(), [dict(id="empty", availability="admitted")] * 3)


def test_unrecognized_observation_is_not_treated_as_admitted():
    with pytest.raises(ValueError, match="availability"):
        account(
            cases(),
            [
                dict(id="empty", availability="admitted"),
                dict(id="small", availability="admitted"),
                dict(id="oversize", availability="timeout"),
            ],
        )


def test_standalone_entry_point_needs_no_site_packages_for_help():
    script = (
        Path(__file__).resolve().parents[1] / "benchmarks/local_memory/local_support_v2_controls.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert "--model-root" in result.stdout


def test_tokenizer_failure_preserves_completed_frames_and_unattempted_denominators(tmp_path):
    from benchmarks.local_memory import local_support_v2_controls as module

    assert hasattr(module, "observe_controls"), "Failure-preserving observation loop is missing"

    class Tokenizer:
        cls_token_id = 2_000_001
        sep_token_id = 2_000_002
        all_special_ids = [cls_token_id, sep_token_id]

        def encode(self, text, *, add_special_tokens):
            if '"query":"crash"' in text:
                raise RuntimeError("Injected tokenizer failure")
            return list(text.encode("utf-8"))

    values = []
    for name in ("first", "second", "third"):
        request = module.request(1)
        if name == "second":
            request["query"] = "crash"
        values.append(
            dict(
                id=name,
                request=request,
                supported_target_ids=[request["candidates"][0]["id"]],
                expected_availability="admitted",
                expected_error_fragment=None,
            )
        )
    with pytest.raises(RuntimeError, match="Injected"):
        module.observe_controls(values, Tokenizer(), "a" * 64, tmp_path)
    failure = json.loads((tmp_path / "failure.json").read_bytes())
    assert failure["status"] == "failed"
    assert failure["failed_control"] == "second"
    assert failure["unattempted_controls"] == ["third"]
    assert failure["expected_controls"] == 3
    assert failure["expected_supported_target_opportunities"] == 3
    assert [v["id"] for v in json.loads((tmp_path / "outcomes.json").read_bytes())] == ["first"]
    assert [v["id"] for v in json.loads((tmp_path / "frames.json").read_bytes())] == ["first"]


def test_terminal_validation_failure_does_not_call_completed_controls_unattempted(
    tmp_path, monkeypatch
):
    from benchmarks.local_memory import local_support_v2_controls as module

    def fail_after_observations(model_root, output):
        module.save(
            output / "outcomes.json",
            [dict(id=identity, availability="admitted") for identity in module.CONTROL_IDS],
        )
        raise RuntimeError("Injected terminal binding failure")

    monkeypatch.setattr(module, "_run", fail_after_observations)
    output = tmp_path / "attempt"
    with pytest.raises(RuntimeError, match="terminal binding"):
        module.run(tmp_path, output)
    failure = json.loads((output / "result.json").read_bytes())
    assert failure["status"] == "failed"
    assert failure["unattempted_controls"] == []
    assert failure["observed_controls"] == 11
    assert failure["expected_supported_target_opportunities"] == 38
