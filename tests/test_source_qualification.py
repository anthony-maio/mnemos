from dataclasses import replace
from datetime import date

import pytest

from benchmarks.local_memory.source_qualification import QualificationCache, QualificationIdentity
from mnemos.answer_support import SupportCandidate, SupportEvidence
from mnemos.support_intersection import SupportSnapshot

IDENTITY = QualificationIdentity("a" * 64, "b" * 64, "c" * 64)


def candidate():
    return SupportCandidate(
        "f1",
        1,
        "East is backed up daily.",
        "project",
        "one",
        (SupportEvidence("e1", "East is backed up daily.", "event1", "source1"),),
        "revision1",
    )


def annotate(cache, *items, identity=IDENTITY, fence=lambda: None):
    return cache.annotate(
        SupportSnapshot("How often?", date(2026, 11, 8), items, "snap"), identity, fence
    )


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"content": "East is backed up hourly."},
        {"fact_id": "f2"},
        {"scope_id": "two"},
        {"scope": "global", "scope_id": None},
        {"fingerprint": "revision2"},
    ],
)
def test_candidate_changes_miss(change):
    cache = QualificationCache()
    original = candidate()
    cache.record(original, "e1", IDENTITY, "supported")
    result = annotate(cache, replace(original, **change))[0]
    assert not result.assessments[0].cache_hit
    assert result.assessments[0].disposition == "unresolved"


@pytest.mark.parametrize(
    "change",
    [
        {"excerpt": "East is backed up hourly."},
        {"source_ref": "other"},
        {"source_event_id": "other"},
        {"evidence_id": "other"},
    ],
)
def test_source_changes_miss(change):
    cache = QualificationCache()
    original = candidate()
    cache.record(original, "e1", IDENTITY, "supported")
    revised = replace(original, evidence=(replace(original.evidence[0], **change),))
    assert not annotate(cache, revised)[0].assessments[0].cache_hit


@pytest.mark.parametrize("field", ["policy_sha256", "model_sha256", "runtime_sha256"])
def test_identity_changes_miss(field):
    cache = QualificationCache()
    original = candidate()
    cache.record(original, "e1", IDENTITY, "supported")
    assert (
        not annotate(cache, original, identity=replace(IDENTITY, **{field: "d" * 64}))[0]
        .assessments[0]
        .cache_hit
    )


def test_preserves_rejected_missing_and_own_source_order():
    cache = QualificationCache()
    first = candidate()
    first = replace(
        first, evidence=first.evidence + (SupportEvidence("e2", "A proposal.", None, None),)
    )
    second = replace(candidate(), fact_id="f2")
    cache.record(first, "e1", IDENTITY, "unsupported")
    result = annotate(cache, first, second)
    assert tuple(r.candidate for r in result) == (first, second)
    assert [a.evidence_id for a in result[0].assessments] == ["e1", "e2"]
    assert result[0].assessments[0].disposition == "unsupported"
    assert not result[0].assessments[1].cache_hit
    assert not result[1].assessments[0].cache_hit
    with pytest.raises(ValueError):
        cache.record(second, "e2", IDENTITY, "supported")


def test_omitted_fact_never_resurrected_and_fence_remains_authoritative():
    cache = QualificationCache()
    original = candidate()
    cache.record(original, "e1", IDENTITY, "supported")
    assert annotate(cache) == ()
    calls = []

    def fence():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("Forgotten or no longer authorized")

    with pytest.raises(RuntimeError, match="Forgotten"):
        annotate(cache, original, fence=fence)
    assert len(calls) == 2


def test_unavailable_cannot_become_negative_and_frozen_assessment_cannot_change():
    cache = QualificationCache()
    original = candidate()
    with pytest.raises(ValueError):
        cache.record(original, "e1", IDENTITY, "unavailable")
    assert not annotate(cache, original)[0].assessments[0].cache_hit
    cache.record(original, "e1", IDENTITY, "unresolved")
    assert annotate(cache, original)[0].assessments[0].cache_hit
    with pytest.raises(ValueError, match="Conflicting"):
        cache.record(original, "e1", IDENTITY, "supported")


@pytest.mark.parametrize("bad", ["", "model-name", "a" * 63, "A" * 64, None])
def test_incomplete_identity_rejected(bad):
    with pytest.raises(ValueError):
        QualificationIdentity(bad, "b" * 64, "c" * 64)


@pytest.mark.parametrize(
    "change", [{"content": 42}, {"version": True}, {"scope_id": 42}, {"fact_id": " "}]
)
def test_malformed_candidate_rejected(change):
    with pytest.raises(ValueError):
        QualificationCache().record(replace(candidate(), **change), "e1", IDENTITY, "supported")


@pytest.mark.parametrize("change", [{"excerpt": 42}, {"source_ref": 42}, {"source_event_id": ""}])
def test_malformed_source_rejected(change):
    original = candidate()
    changed = replace(original, evidence=(replace(original.evidence[0], **change),))
    with pytest.raises(ValueError):
        QualificationCache().record(changed, "e1", IDENTITY, "supported")
