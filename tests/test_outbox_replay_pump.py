import asyncio

import pytest

from mnemos.client import MnemosClient, Outbox
from mnemos.outbox_replay import OutboxReplayPump


class Client:
    def __init__(self):
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.closed = False

    def begin_host_event(self):
        pass

    async def replay_outbox(self):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"acknowledged": 1, "failed": 0, "pending": 0}

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_start_drains_once_and_wake_during_delivery_is_not_lost():
    client = Client()
    pump = OutboxReplayPump(client, retry_seconds=60)
    pump.start()
    pump.start()
    await asyncio.wait_for(client.started.wait(), 1)
    for _ in range(10):
        pump.notify()
    client.release.set()
    for _ in range(100):
        if client.calls == 2 and pump.last_result is not None:
            break
        await asyncio.sleep(0.001)
    assert client.calls == 2
    await pump.close()
    assert client.closed
    with pytest.raises(RuntimeError):
        pump.notify()
    with pytest.raises(RuntimeError):
        pump.start()


@pytest.mark.asyncio
async def test_close_cancels_active_delivery_before_closing_client():
    client = Client()
    pump = OutboxReplayPump(client)
    pump.start()
    await asyncio.wait_for(client.started.wait(), 1)
    await pump.close()
    await pump.close()
    assert client.cancelled and client.closed
    assert pump.last_result is None


@pytest.mark.asyncio
async def test_unavailable_service_leaves_disk_queue_across_pump_restart(tmp_path):
    path = tmp_path / "outbox.json"
    Outbox(path).append({"event_id": "original", "content": "Retain this event."})
    for _ in range(2):
        client = MnemosClient(
            descriptor_path=tmp_path / "runtime.json", outbox=Outbox(path), auto_start=False
        )
        pump = OutboxReplayPump(client, retry_seconds=60)
        pump.start()
        for _ in range(100):
            if pump.last_result is not None:
                break
            await asyncio.sleep(0.01)
        assert pump.last_result == {
            "schema_version": 1,
            "acknowledged": 0,
            "failed": 1,
            "pending": 1,
        }
        await pump.close()
        assert Outbox(path).pending()[0]["payload"]["event_id"] == "original"


@pytest.mark.asyncio
async def test_exception_retries_without_becoming_success():
    class Failing(Client):
        async def replay_outbox(self):
            self.calls += 1
            raise ValueError("unreadable queue")

    client = Failing()
    pump = OutboxReplayPump(client, retry_seconds=0.02)
    pump.start()
    for _ in range(100):
        if client.calls >= 2:
            break
        await asyncio.sleep(0.005)
    assert client.calls >= 2 and pump.last_result is None
    assert pump.last_error_type == "ValueError"
    await pump.close()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_retry_interval(interval):
    with pytest.raises(ValueError):
        OutboxReplayPump(None, retry_seconds=interval)
