# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

_PROXY_SERVER = (
    Path(__file__).parents[2]
    / "examples"
    / "disaggregated"
    / "disaggregated_serving"
    / "moriio_toy_proxy_server.py"
)


class _QuartStub:
    def __init__(self, *args, **kwargs):
        pass

    def route(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func

    def run(self, *args, **kwargs):
        pass


class _RequestStub:
    async def get_json(self):
        return {"max_tokens": 2}


async def _make_response_stub(response):
    return response


def _install_import_stubs(monkeypatch):
    quart = ModuleType("quart")
    quart.Quart = _QuartStub
    quart.Request = _RequestStub
    quart.make_response = _make_response_stub
    quart.request = _RequestStub()
    monkeypatch.setitem(sys.modules, "quart", quart)

    aiohttp = ModuleType("aiohttp")
    aiohttp.ClientSession = object
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "msgpack", ModuleType("msgpack"))
    monkeypatch.setitem(sys.modules, "zmq", ModuleType("zmq"))

    moriio_common = ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
    )
    moriio_common.MoRIIOConstants = SimpleNamespace(TRANSFER_PREFIX="transfer")
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common",
        moriio_common,
    )


def _load_proxy_module(monkeypatch):
    _install_import_stubs(monkeypatch)
    module_name = "_moriio_toy_proxy_server_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _PROXY_SERVER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefill_dp_rank_rotation_advances_per_prefill_endpoint(monkeypatch):
    proxy = _load_proxy_module(monkeypatch)
    num_prefill_instances = 2
    dp_size = 2
    selected_ranks_by_prefill = [[] for _ in range(num_prefill_instances)]

    for request_number in range(1, 1 + num_prefill_instances * dp_size * 2):
        prefill_index = (request_number - 1) % num_prefill_instances
        prefill_request_number = proxy.example_prefill_request_number(
            request_number, num_prefill_instances
        )
        selected_rank = proxy.example_round_robin_dp_loader(
            prefill_request_number, dp_size
        )
        selected_ranks_by_prefill[prefill_index].append(selected_rank)

    assert selected_ranks_by_prefill == [[0, 1, 0, 1], [0, 1, 0, 1]]


def test_handle_request_uses_prefill_local_request_number_for_dp_rank(monkeypatch):
    proxy = _load_proxy_module(monkeypatch)
    proxy.TRANSFER_TYPE = "READ"
    proxy.request_nums = 0
    proxy.prefill_instances[:] = [
        {
            "request_address": "http://prefill-0",
            "zmq_address": "prefill-zmq-0",
            "dp_size": 2,
            "tp_size": 1,
        },
        {
            "request_address": "http://prefill-1",
            "zmq_address": "prefill-zmq-1",
            "dp_size": 2,
            "tp_size": 1,
        },
    ]
    proxy.decode_instances[:] = [
        {
            "request_address": "http://decode-0",
            "zmq_address": "decode-zmq-0",
            "dp_size": 1,
            "tp_size": 1,
        }
    ]
    selected_prefill_ranks = []

    async def send_request_to_prefill(endpoint, req_data, request_id, selected_rank):
        selected_prefill_ranks.append((endpoint, selected_rank))
        return {
            "kv_transfer_params": {
                "remote_engine_id": 1,
                "remote_block_ids": [2],
                "transfer_id": req_data["kv_transfer_params"]["transfer_id"],
            }
        }

    async def start_decode_request(endpoint, req_data, request_id):
        return object(), object()

    monkeypatch.setattr(proxy, "send_request_to_prefill", send_request_to_prefill)
    monkeypatch.setattr(proxy, "start_decode_request", start_decode_request)
    monkeypatch.setattr(proxy, "stream_decode_response", lambda *args: b"")

    for _ in range(4):
        asyncio.run(proxy.handle_request("/completions", _RequestStub()))

    assert selected_prefill_ranks == [
        ("http://prefill-0/completions", 0),
        ("http://prefill-1/completions", 0),
        ("http://prefill-0/completions", 1),
        ("http://prefill-1/completions", 1),
    ]
