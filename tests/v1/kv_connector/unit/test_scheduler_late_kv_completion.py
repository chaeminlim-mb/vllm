import time
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus


class DummyConnector:
    def __init__(self) -> None:
        self.outputs: list[KVConnectorOutput] = []
        self.pending_deferred_sends = False

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        self.outputs.append(output)

    def has_pending_deferred_sends(self) -> bool:
        return self.pending_deferred_sends

    def take_events(self):
        return None


class DummyKVCacheManager:
    def take_events(self):
        return None


class DummyKVEventPublisher:
    def __init__(self) -> None:
        self.batches: list[object] = []

    def publish(self, batch: object) -> None:
        self.batches.append(batch)


class DummyBlocks:
    def __init__(self, block_ids: list[int]) -> None:
        self._block_ids = block_ids

    def get_block_ids(self) -> tuple[list[int], ...]:
        return (self._block_ids,)


def make_scheduler(requests: dict[str, object] | None = None) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = DummyConnector()
    scheduler.requests = requests or {}
    scheduler.finished_req_ids = set()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.perf_metrics = None
    scheduler.kv_cache_manager = DummyKVCacheManager()
    scheduler.kv_event_publisher = DummyKVEventPublisher()
    scheduler.make_stats = lambda *_args: None
    return scheduler


def make_moriio_scheduler(block_size: int = 16) -> MoRIIOConnectorScheduler:
    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler.block_size = block_size
    return scheduler


def test_late_finished_sending_for_removed_request_is_ignored() -> None:
    scheduler = make_scheduler()

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={"already-finished"})
    )

    assert scheduler.requests == {}
    assert len(scheduler.connector.outputs) == 1


def test_late_finished_recving_for_removed_request_is_ignored() -> None:
    scheduler = make_scheduler()

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"already-aborted"})
    )

    assert scheduler.finished_recving_kv_req_ids == set()
    assert len(scheduler.connector.outputs) == 1


def test_live_finished_sending_still_frees_blocks() -> None:
    request = SimpleNamespace(request_id="live")
    scheduler = make_scheduler({"live": request})
    freed: list[object] = []
    scheduler._free_blocks = freed.append

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={"live"})
    )

    assert freed == [request]


def test_live_finished_recving_still_marks_waiting_request() -> None:
    request = SimpleNamespace(status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    scheduler = make_scheduler({"live": request})

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"live"})
    )

    assert scheduler.finished_recving_kv_req_ids == {"live"}


def test_pending_deferred_send_keeps_scheduler_active() -> None:
    scheduler = make_scheduler()
    assert not scheduler.has_finished_requests()

    scheduler.connector.pending_deferred_sends = True

    assert scheduler.has_finished_requests()


def test_pending_deferred_send_updates_connector_without_worker_output() -> None:
    scheduler = make_scheduler()
    scheduler.connector.pending_deferred_sends = True

    scheduler.update_from_output(
        SimpleNamespace(num_scheduled_tokens={}),
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert len(scheduler.connector.outputs) == 1
    assert scheduler.connector.outputs[0].is_empty()


def test_deferred_send_stays_active_between_grace_and_timeout() -> None:
    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler._deferred_send_deadlines = {"req": time.monotonic() + 60.0}
    scheduler._deferred_send_drain_until = time.monotonic() - 1.0

    assert scheduler.has_pending_deferred_sends()


def test_moriio_read_transfer_uses_cached_suffix_blocks() -> None:
    scheduler = make_moriio_scheduler()
    request = SimpleNamespace(
        request_id="req",
        num_tokens=65,
        prompt_token_ids=list(range(65)),
    )

    local_block_ids, remote_block_ids = scheduler._select_read_transfer_blocks(
        request,
        DummyBlocks([10, 11, 12, 13, 14]),
        [2000, 2001, 2002, 2003, 2004],
        num_external_tokens=32,
    )

    assert local_block_ids == [12, 13]
    assert remote_block_ids == [2002, 2003]


def test_moriio_read_transfer_ignores_lookahead_blocks() -> None:
    scheduler = make_moriio_scheduler()
    request = SimpleNamespace(
        request_id="req",
        num_tokens=33,
        prompt_token_ids=list(range(33)),
    )

    local_block_ids, remote_block_ids = scheduler._select_read_transfer_blocks(
        request,
        DummyBlocks([10, 11, 12, 99]),
        [2000, 2001, 2002],
        num_external_tokens=32,
    )

    assert local_block_ids == [10, 11]
    assert remote_block_ids == [2000, 2001]


def test_moriio_read_transfer_excludes_trailing_local_block() -> None:
    scheduler = make_moriio_scheduler()
    request = SimpleNamespace(
        request_id="req",
        num_tokens=33,
        prompt_token_ids=list(range(33)),
    )

    local_block_ids, remote_block_ids = scheduler._select_read_transfer_blocks(
        request,
        DummyBlocks([10, 11, 12]),
        [2000, 2001, 2002],
        num_external_tokens=32,
    )

    assert local_block_ids == [10, 11]
    assert remote_block_ids == [2000, 2001]


def test_moriio_read_transfer_returns_empty_blocks_for_full_prefix_hit() -> None:
    scheduler = make_moriio_scheduler()
    request = SimpleNamespace(
        request_id="req",
        num_tokens=64,
        prompt_token_ids=list(range(64)),
    )

    local_block_ids, remote_block_ids = scheduler._select_read_transfer_blocks(
        request,
        DummyBlocks([10, 11, 12, 13]),
        [2000, 2001],
        num_external_tokens=0,
    )

    assert local_block_ids == []
    assert remote_block_ids == []


def test_moriio_read_transfer_rejects_partial_external_blocks() -> None:
    scheduler = make_moriio_scheduler()
    request = SimpleNamespace(
        request_id="req",
        num_tokens=49,
        prompt_token_ids=list(range(49)),
    )

    with pytest.raises(RuntimeError, match="external block count 3 exceeds local"):
        scheduler._select_read_transfer_blocks(
            request,
            DummyBlocks([10, 11]),
            [2000, 2001],
            num_external_tokens=48,
        )
