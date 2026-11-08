"""Concurrent support must finish or cancel every provider before returning."""

import asyncio

import pytest

from mnemos.support_intersection import IntersectionConfig
from mnemos.utils.reliability import MnemosConfigurationError, MnemosTransientError
from tests.test_answer_support import fact, memory  # noqa: F401
from tests.test_support_intersection import Selector, assess


@pytest.mark.asyncio
@pytest.mark.parametrize("selections", [((), ()), (("one",), ()), (("one",), ("one",))])
async def test_execution_mode_preserves_decisions_and_evidence(memory, selections):
    fact(memory, "one")
    results = []
    for mode in ("sequential", "concurrent"):
        results.append(
            await assess(
                memory,
                Selector(selections[0]),
                Selector(selections[1]),
                ids=("one",),
                config=IntersectionConfig(execution=mode),
            )
        )
    assert results[0] == results[1]


@pytest.mark.asyncio
async def test_concurrent_selectors_overlap_and_preserve_component_order(memory):
    fact(memory, "one")
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def first_action(_):
        entered[0].set()
        await entered[1].wait()
        await release.wait()

    async def second_action(_):
        entered[1].set()
        await entered[0].wait()
        release.set()

    first, second = Selector(("one",), first_action), Selector((), second_action)
    result = await asyncio.wait_for(
        assess(
            memory, first, second, ids=("one",), config=IntersectionConfig(execution="concurrent")
        ),
        timeout=2,
    )
    assert result.status == "assessed" and not result.supported
    assert result.components[0].selected[0].fact_id == "one"
    assert result.components[1].selected == ()
    assert first.calls[0] is second.calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("outcome", ["provider", "configuration", "stale", "timeout", "cancel"])
async def test_concurrent_failure_or_cancellation_drains_both_selectors(memory, stage, outcome):
    fact(memory, "one")
    entered = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]

    def action(index):
        async def run(_):
            entered[index].set()
            try:
                await entered[1 - index].wait()
                if index == stage:
                    if outcome == "provider":
                        raise MnemosTransientError("offline")
                    if outcome == "configuration":
                        raise MnemosConfigurationError("missing model")
                    if outcome == "stale":
                        memory[0].delete("one")
                        return
                await asyncio.Event().wait()
            finally:
                finished[index].set()

        return run

    task = asyncio.create_task(
        assess(
            memory,
            Selector(("one",), action(0)),
            Selector(("one",), action(1)),
            ids=("one",),
            config=IntersectionConfig(
                execution="concurrent", timeout_seconds=0.1 if outcome == "timeout" else 2
            ),
        )
    )
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), timeout=2)
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "configuration":
        with pytest.raises(MnemosConfigurationError):
            await task
    else:
        result = await task
        assert not result.supported
        assert result.status == ("partial" if outcome == "stale" else "unavailable")
    assert all(event.is_set() for event in finished)
