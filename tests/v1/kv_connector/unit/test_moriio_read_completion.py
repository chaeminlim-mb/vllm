# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorWorker,
)


class FakeStatus:
    def __init__(
        self,
        *,
        succeeded: bool = False,
        failed: bool = False,
        succeed_after: int = 0,
    ) -> None:
        self.succeeded = succeeded
        self.failed = failed
        self.succeed_after = succeed_after
        self.succeeded_calls = 0

    def Succeeded(self) -> bool:
        self.succeeded_calls += 1
        if self.succeed_after and self.succeeded_calls >= self.succeed_after:
            self.succeeded = True
        return self.succeeded

    def Failed(self) -> bool:
        return self.failed

    def Message(self) -> str:
        return "fake failure"

    def Code(self) -> int:
        return 123


class FakeWrapper:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.notifies: list[tuple[str, str, str]] = []

    def send_notify(self, transfer_id: str, host: str, port: str) -> None:
        self.notifies.append((transfer_id, host, port))

    def shutdown(self) -> None:
        pass


def make_worker() -> MoRIIOConnectorWorker:
    worker = object.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.READ
    worker.moriio_config = SimpleNamespace(transfer_timeout=1.0)
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = defaultdict(dict)
    worker._recving_transfers_callback_addr = {}
    worker.transfer_id_to_request_id = {}
    return worker


def test_finished_count_tracks_tensor_parallel_size() -> None:
    connector = object.__new__(MoRIIOConnector)
    connector._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=8)
    )

    assert connector.get_finished_count() == 1

    connector._vllm_config.parallel_config.tensor_parallel_size = 8
    connector._vllm_config.parallel_config.data_parallel_size = 1

    assert connector.get_finished_count() == 8


def test_wait_for_layer_load_waits_until_layer_status_succeeds() -> None:
    worker = make_worker()
    status = FakeStatus(succeed_after=3)
    worker._recving_transfers["req0"]["layer0"] = status

    worker.wait_for_layer_load("layer0")

    assert status.succeeded_calls >= 3


def test_wait_for_layer_load_raises_on_failed_status() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(failed=True)

    with pytest.raises(RuntimeError, match="request req0, layer layer0"):
        worker.wait_for_layer_load("layer0")


def test_pop_done_transfers_waits_for_all_layer_statuses() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(succeeded=True)
    worker._recving_transfers["req0"]["layer1"] = FakeStatus()
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker.transfer_id_to_request_id["transfer0"] = "req0"

    assert worker._pop_done_transfers() == set()
    assert worker.moriio_wrapper.notifies == []
    assert "req0" in worker._recving_transfers

    worker._recving_transfers["req0"]["layer1"].succeeded = True

    assert worker._pop_done_transfers() == set()
    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "1234")]
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "transfer0" not in worker.transfer_id_to_request_id
