# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import copy
import json
import logging
import os
import socket
import threading
import uuid
from urllib.parse import urlparse

import aiohttp
import msgpack
import zmq
from quart import Quart, Request, make_response, request

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConstants,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
prefill_instances: list[dict] = []
decode_instances: list[dict] = []
request_nums = 0
app = Quart(__name__)


TRANSFER_TYPE = None


_list_lock = threading.RLock()


def _listen_for_register(hostname, port):
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    global prefill_instances
    global decode_instances

    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_addr, msg = router_socket.recv_multipart()
            data = msgpack.loads(msg)
            if data.get("type") == "HELLO":
                pass
            elif data.get("type") in ("P", "D"):
                role = data["type"]
                required_keys = {
                    "http_address",
                    "zmq_address",
                    "dp_size",
                    "tp_size",
                    "transfer_mode",
                }
                missing = required_keys - data.keys()
                if missing:
                    logger.error(
                        "Registration message missing required keys %s; skipping",
                        missing,
                    )
                    continue
                # Derive request_address from http_address
                # api path suffix is appended at request time
                instance = {
                    "role": role,
                    "request_address": f"http://{data['http_address']}/v1",
                    "http_address": data["http_address"],
                    "zmq_address": data["zmq_address"],
                    "dp_size": data["dp_size"],
                    "tp_size": data["tp_size"],
                    "transfer_mode": data["transfer_mode"],
                    "node_hosts": data.get("node_hosts", []),
                }
                # zmq_address format: "host:IP,handshake:PORT,notify:PORT"
                # Stored verbatim; embedded into the request_id by handle_request.

                global TRANSFER_TYPE
                transfer_mode = instance["transfer_mode"]
                target_list = prefill_instances if role == "P" else decode_instances
                with _list_lock:
                    if TRANSFER_TYPE is None:
                        TRANSFER_TYPE = transfer_mode
                        logger.info("SET TRANSFER TYPE TO %s", TRANSFER_TYPE)
                    elif transfer_mode != TRANSFER_TYPE:
                        logger.error(
                            "Mismatched transfer mode: expected %s, got %s;"
                            " skipping registration of %s",
                            TRANSFER_TYPE,
                            transfer_mode,
                            data["http_address"],
                        )
                        continue
                    existing_idx = next(
                        (
                            idx
                            for idx, i in enumerate(target_list)
                            if i.get("http_address") == data["http_address"]
                        ),
                        None,
                    )
                    if existing_idx is not None:
                        target_list[existing_idx] = instance
                        logger.info(
                            "Updated existing %s instance: %s",
                            "Prefill" if role == "P" else "Decode",
                            instance,
                        )
                    else:
                        target_list.append(instance)
                        logger.info(
                            "Registered %s instance: %s",
                            "Prefill" if role == "P" else "Decode",
                            instance,
                        )
            else:
                logger.warning(
                    "Received message with unrecognized type %r; ignoring",
                    data.get("type"),
                )


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=(hostname, port), daemon=True
    )
    _listener_thread.start()
    return _listener_thread


async def send_request_to_prefill(
    endpoint, req_data, request_id, selected_prefill_dp_rank
):
    req_data_copy = req_data

    req_data_copy["kv_transfer_params"].update(
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
        }
    )
    req_data_copy["stream"] = False
    req_data_copy["max_tokens"] = 1
    if "max_completion_tokens" in req_data_copy:
        req_data_copy["max_completion_tokens"] = 1
    if "stream_options" in req_data_copy:
        del req_data_copy["stream_options"]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    ) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        if selected_prefill_dp_rank is not None:
            headers["X-data-parallel-rank"] = str(selected_prefill_dp_rank)
        async with session.post(
            url=endpoint, json=req_data_copy, headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()

            else:
                error_message = (
                    f"send_request_to_prefill response ={response},"
                    f"reason={response.reason}, status={response.status},"
                    f"method={response.method}, url={response.url},"
                    f"real_url={response.real_url}"
                )
                raise RuntimeError(error_message)


async def start_decode_request(endpoint, req_data, request_id, data_parallel_rank=None):
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    )
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
    response = await session.post(url=endpoint, json=req_data, headers=headers)
    return session, response


async def stream_decode_response(session, response, request_id):
    try:
        if response.status == 200:
            async for chunk_bytes in response.content.iter_chunked(1024):
                yield chunk_bytes
        else:
            error_message = (
                f"stream_decode_response response ={response},"
                f"reason={response.reason}, status={response.status},"
                f"method={response.method}, url={response.url},"
                f"real_url={response.real_url}"
            )
            raise RuntimeError(error_message)
    finally:
        await session.close()


def example_round_robin_dp_loader(request_number, dp_size):
    return (request_number - 1) % dp_size


@app.route("/health", methods=["GET"])
async def health():
    # exp_common.sh:971 post-bench wait_health probes the PROXY url (not the
    # vllm engine), and treats any non-200 as "SERVER DIED" → kills bench
    # mid-sweep even when prefill+decode are healthy. Return 200 unconditionally.
    return ("ok", 200)


@app.route("/v1/completions", methods=["POST"])
async def handle_completions_request():
    return await handle_request("/completions", request)


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_chat_completions_request():
    return await handle_request("/chat/completions", request)


async def handle_request(api: str, request: Request):
    try:
        with _list_lock:
            global request_nums
            request_nums += 1
            request_number = request_nums

        req_data = await request.get_json()

        prefill_instance_endpoint = None
        decode_instance_endpoint = None
        error_msg = (
            "Service Unavailable: No prefill or decode instances are registered."
        )
        if not prefill_instances or not decode_instances:
            return await make_response(
                (
                    error_msg,
                    503,
                )
            )
        pid = (request_number - 1) % len(prefill_instances)
        did = (request_number - 1) % len(decode_instances)
        prefill_instance_endpoint = prefill_instances[pid]
        decode_instance_endpoint = decode_instances[did]

        selected_prefill_dp_rank = None
        if prefill_instance_endpoint["dp_size"] > 1:
            selected_prefill_dp_rank = example_round_robin_dp_loader(
                request_number,
                prefill_instance_endpoint["dp_size"],
            )

        # Decode routes to its OWN dp rank, derived from the DECODE's dp_size —
        # not the prefill's. With heterogeneous parallelism (e.g. DP8EP prefill +
        # TP8 decode, dp_size=1), forwarding the prefill rank yields
        # "data_parallel_rank N out of range [0,1)" -> 400 on the decode. For
        # uniform DP8<->DP8 this matches the prefill rank (same loader/args), so
        # behavior is unchanged there.
        selected_decode_dp_rank = None
        if decode_instance_endpoint["dp_size"] > 1:
            selected_decode_dp_rank = example_round_robin_dp_loader(
                request_number,
                decode_instance_endpoint["dp_size"],
            )

        # Embed both zmq_addresses in the request_id so the connector can parse
        # the peer's host/ports from it, similar to P2P-NCCL
        uid = str(uuid.uuid4()).replace("-", "")
        request_id = (
            f"___prefill_addr_{prefill_instance_endpoint['zmq_address']}"
            f"___decode_addr_{decode_instance_endpoint['zmq_address']}"
            f"_{uid}"
        )

        transfer_id = f"{MoRIIOConstants.TRANSFER_PREFIX}-{str(uuid.uuid4())}"

        req_data_to_prefill = copy.deepcopy(req_data)
        req_data_to_prefill["kv_transfer_params"] = {}
        req_data["kv_transfer_params"] = {}
        req_data_to_prefill["kv_transfer_params"]["remote_dp_size"] = (
            decode_instance_endpoint["dp_size"]
        )
        req_data_to_prefill["kv_transfer_params"]["remote_tp_size"] = (
            decode_instance_endpoint["tp_size"]
        )
        req_data_to_prefill["kv_transfer_params"]["transfer_id"] = transfer_id
        decode_hosts = decode_instance_endpoint.get("node_hosts") or []
        if decode_hosts:
            req_data_to_prefill["kv_transfer_params"]["remote_hosts"] = list(
                decode_hosts
            )
        if selected_prefill_dp_rank is not None:
            req_data_to_prefill["kv_transfer_params"]["remote_dp_rank"] = (
                selected_prefill_dp_rank
            )

        prefill_request_url = prefill_instance_endpoint["request_address"] + api
        send_prefill_task = asyncio.create_task(
            send_request_to_prefill(
                prefill_request_url,
                req_data_to_prefill,
                request_id,
                selected_prefill_dp_rank,
            )
        )

        # OpenAI Chat Completions: when both fields are present,
        # max_completion_tokens takes precedence, so decrement that one
        # to keep the per-request budget consistent with what the
        # backend will enforce. Fall back to max_tokens otherwise.
        if "max_completion_tokens" in req_data:
            req_data["max_completion_tokens"] -= 1
        elif "max_tokens" in req_data:
            req_data["max_tokens"] -= 1

        req_data["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "transfer_id": transfer_id,
        }
        if TRANSFER_TYPE == "READ":
            # In read mode, prefill and decode are executed serially.
            prefill_response = await send_prefill_task
            prefill_kv = prefill_response["kv_transfer_params"]
            req_data["kv_transfer_params"]["remote_engine_id"] = prefill_kv[
                "remote_engine_id"
            ]
            req_data["kv_transfer_params"]["remote_block_ids"] = prefill_kv[
                "remote_block_ids"
            ]
            req_data["kv_transfer_params"]["transfer_id"] = prefill_kv["transfer_id"]
            req_data["kv_transfer_params"]["remote_hosts"] = prefill_kv.get(
                "remote_hosts"
            )
            # Forward P-side wall-clock prefill-completion stamp so D-side
            # PD breakdown sees stage-3 cross-node sanity check.
            prefill_complete_ts = prefill_kv.get("prefill_complete_ts")
            if prefill_complete_ts is not None:
                req_data["kv_transfer_params"]["prefill_complete_ts"] = (
                    prefill_complete_ts
                )
        prefill_hosts = prefill_instance_endpoint.get("node_hosts") or []
        if prefill_hosts:
            req_data["kv_transfer_params"]["remote_hosts"] = list(prefill_hosts)

        req_data["kv_transfer_params"]["remote_dp_size"] = prefill_instance_endpoint[
            "dp_size"
        ]
        req_data["kv_transfer_params"]["remote_tp_size"] = prefill_instance_endpoint[
            "tp_size"
        ]
        req_data["kv_transfer_params"]["tp_size"] = prefill_instance_endpoint["tp_size"]

        if selected_prefill_dp_rank is not None:
            req_data["kv_transfer_params"]["remote_dp_rank"] = selected_prefill_dp_rank

        decode_request_url = decode_instance_endpoint["request_address"] + api
        decode_request_task = asyncio.create_task(
            start_decode_request(
                decode_request_url,
                req_data,
                request_id,
                selected_decode_dp_rank,
            )
        )

        session, decode_response = await decode_request_task
        stream_generator = stream_decode_response(session, decode_response, request_id)
        response = await make_response(stream_generator)
        return response
    except Exception as e:
        logger.exception("An error occurred while handling the request: %s", e)
        return await make_response(
            (
                f"Internal Server Error: {e!s}",
                500,
            )
        )


# ── Profiler fan-out ────────────────────────────────────────────────────────
# benchmark_serving.py --profile sends POST /start_profile (and /stop_profile)
# to the base URL it was given. For PD that base URL is the proxy, so we
# forward the request to every registered prefill + decode backend so torch
# traces are captured on both sides of the disaggregation for the same window.


async def _post_profile_to_endpoint(session, base_url, cmd, payload):
    url = f"{base_url}/{cmd}_profile"
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 404:
                logger.warning("Profiling endpoint missing on %s", url)
                return None
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"{url} returned {resp.status}: {body}")
            # vLLM's /start_profile and /stop_profile return plain text with no
            # JSON content-type header. content_type=None bypasses aiohttp's
            # mimetype check; on parse failure fall back to a synthetic envelope.
            try:
                return await resp.json(content_type=None)
            except Exception:
                return {"status": "ok", "raw": await resp.text()}
    except aiohttp.ClientResponseError as exc:
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise


def _collect_backend_base_urls():
    with _list_lock:
        p_urls = []
        for inst in prefill_instances:
            _p = urlparse(inst["request_address"])
            p_urls.append(f"http://{_p.hostname}:{_p.port}")
        d_urls = []
        for inst in decode_instances:
            _p = urlparse(inst["request_address"])
            d_urls.append(f"http://{_p.hostname}:{_p.port}")
    return p_urls, d_urls


async def _handle_profile_command(cmd):
    payload = await request.get_json(silent=True) or {}
    p_urls, d_urls = _collect_backend_base_urls()
    if not p_urls and not d_urls:
        return await make_response(
            (
                json.dumps({"error": "no prefill or decode backends registered"}),
                503,
                {"Content-Type": "application/json"},
            )
        )

    # /stop_profile flush time scales with concurrency × profile metadata
    # (record_shapes / with_stack add GBs of per-event data). For heavy
    # profile runs (c=100+ with full annotations) the worker stop_profile
    # RPC can block for 10–20 minutes while torch exports the trace and
    # writes profiler_out_N.txt. Match the chat-completions timeout window
    # so we don't return 500 before the backend finishes flushing.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=3600)
    ) as session:
        p_results, d_results = await asyncio.gather(
            asyncio.gather(*[
                _post_profile_to_endpoint(session, u, cmd, payload) for u in p_urls
            ]),
            asyncio.gather(*[
                _post_profile_to_endpoint(session, u, cmd, payload) for u in d_urls
            ]),
        )

    return {
        "prefill": [{"url": u, "result": r} for u, r in zip(p_urls, p_results)],
        "decode": [{"url": u, "result": r} for u, r in zip(d_urls, d_results)],
    }


@app.post("/start_profile")
async def start_profile():
    return await _handle_profile_command("start")


@app.post("/stop_profile")
async def stop_profile():
    return await _handle_profile_command("stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    args = parser.parse_args()

    t = start_service_discovery("0.0.0.0", 36367)
    app.debug = True
    app.config["BODY_TIMEOUT"] = 360000
    app.config["RESPONSE_TIMEOUT"] = 360000

    app.run(host="0.0.0.0", port=args.port)
    t.join()
