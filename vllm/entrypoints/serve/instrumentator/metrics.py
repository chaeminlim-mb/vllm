# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.routing import Mount

from vllm.v1.metrics.prometheus import get_prometheus_registry

# vLLM's modular include_router (chat/completion/scoring/generate/...) leaves
# fastapi _IncludedRouter sentinels in app.routes. These have no `.path`, and
# prometheus-fastapi-instrumentator 8.0.0 calls route.path on every route for
# every request -> AttributeError -> HTTP 500 on ALL endpoints (incl. /health,
# /v1/models). Guard the route-name lookup so path-less routes are skipped
# instead of crashing the request.
import prometheus_fastapi_instrumentator.routing as _pfi_routing

_pfi_orig_get_route_name = _pfi_routing._get_route_name


def _pfi_safe_get_route_name(scope, routes):
    try:
        return _pfi_orig_get_route_name(scope, routes)
    except AttributeError:
        return None


_pfi_routing._get_route_name = _pfi_safe_get_route_name


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()

    # HTTP-request instrumentation is OFF by default. vLLM's modular routers
    # leave fastapi _IncludedRouter sentinels (no .path) in app.routes, which
    # crash prometheus-fastapi-instrumentator 8.0.0 on EVERY request -> HTTP
    # 500 on all endpoints (incl. /health, /v1/models). Opt in with
    # VLLM_ENABLE_FASTAPI_METRICS=1 (the routing guard above keeps it from
    # crashing even when enabled). The engine /metrics endpoint below is
    # independent and always mounted.
    if os.getenv("VLLM_ENABLE_FASTAPI_METRICS", "0") == "1":
        # response_class=PrometheusResponse returns Content-Type
        # "text/plain; version=0.0.4" instead of the default JSON.
        Instrumentator(
            excluded_handlers=[
                "/metrics",
                "/health",
                "/load",
                "/ping",
                "/version",
                "/server_info",
            ],
            registry=registry,
        ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
