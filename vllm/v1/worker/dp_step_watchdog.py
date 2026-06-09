# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step worker watchdog (diagnostic for the wideEP DP all_reduce wedge).

Background / motivation
-----------------------
Under load, a few DP ranks of a multi-rank prefill instance can stall on the
per-step worker execution path *before* the per-step DP-coordination
all_reduce (gpu_model_runner.execute_model / _dummy_run ->
_determine_batch_execution_and_padding -> coordinate_batch_across_dp ->
_synchronize_dp_ranks -> dp_utils._run_ar -> dist.all_reduce(...)).

Because NCCL-for-DP is disabled under async scheduling, that DP group is a GLOO
CPU process group whose collective timeout defaults to torch's 1800s. One
rank's per-step stall therefore becomes an instance-wide 16-way 30-minute
all_reduce deadlock. The exact reason a rank stalls *before* reaching _run_ar
is not yet known, so this watchdog captures the blocked frame on the next
reproduction.

Design
------
A single background daemon thread compares "now" against a per-step heartbeat
timestamp that the worker updates at the tightest per-step point that *every*
DP rank passes each step (including idle ranks running dummy batches). If the
heartbeat is older than the threshold, the watchdog dumps *all* Python thread
tracebacks (faulthandler.dump_traceback(all_threads=True)) to sys.stderr (so it
lands in docker logs) exactly once per stall episode, then re-arms after the
heartbeat advances again. This captures a rank stalled *before* it reaches the
all_reduce, which a collective timeout alone cannot reveal.

Gating
------
Entirely disabled unless VLLM_DP_STEP_WATCHDOG_S is set to a positive number of
seconds (e.g. 60). When disabled, heartbeat() is a couple of cheap branches and
no background thread is ever started, so overhead is negligible.
"""

import faulthandler
import os
import sys
import threading
import time

_ENV_VAR = "VLLM_DP_STEP_WATCHDOG_S"


def _read_threshold_seconds() -> float | None:
    """Return the configured stall threshold in seconds, or None if disabled."""
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class _DPStepWatchdog:
    """Heartbeat-driven watchdog with a single background daemon thread."""

    def __init__(self, threshold_seconds: float) -> None:
        self._threshold = threshold_seconds
        # Monotonic timestamp of the most recent per-step heartbeat.
        self._last_beat = time.monotonic()
        # True while we are inside a step (between enter() and exit()). The
        # watchdog only fires for an actively-running step that makes no
        # progress, not for an idle worker blocked waiting for the next RPC.
        self._in_step = False
        # Whether we have already dumped tracebacks for the current stall
        # episode; reset (re-armed) once the heartbeat advances again.
        self._dumped = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="dp-step-watchdog",
            daemon=True,
        )
        self._thread.start()

    def beat(self) -> None:
        """Mark forward progress at a per-step point and (re)arm the watchdog."""
        with self._lock:
            self._last_beat = time.monotonic()
            self._in_step = True
            self._dumped = False

    def idle(self) -> None:
        """Mark the worker as between steps so we don't fire while idle."""
        with self._lock:
            self._in_step = False
            self._last_beat = time.monotonic()
            self._dumped = False

    def _run(self) -> None:
        # Poll at a fraction of the threshold so detection latency is bounded
        # without busy-spinning.
        poll_interval = max(self._threshold / 4.0, 1.0)
        while True:
            time.sleep(poll_interval)
            with self._lock:
                in_step = self._in_step
                stalled = (time.monotonic() - self._last_beat) > self._threshold
                already_dumped = self._dumped
                if in_step and stalled and not already_dumped:
                    self._dumped = True
                    should_dump = True
                else:
                    should_dump = False
            if should_dump:
                self._dump()

    def _dump(self) -> None:
        # Dump every Python thread's traceback to stderr so it lands in docker
        # logs. This is the key signal for the wideEP DP all_reduce wedge: it
        # captures ranks stalled *before* reaching dp_utils._run_ar's
        # dist.all_reduce, not only ranks already blocked inside the collective.
        try:
            print(
                f"\n[dp-step-watchdog] no per-step progress for "
                f">{self._threshold:.0f}s (pid={os.getpid()}); dumping all "
                f"thread tracebacks (wideEP DP all_reduce wedge diagnostic):",
                file=sys.stderr,
                flush=True,
            )
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            sys.stderr.flush()
        except Exception:
            # The watchdog must never crash the worker.
            pass


# Module-level lazy singleton. None until first use; False once we have decided
# the watchdog is disabled (so we never re-read the env on the hot path).
_watchdog: "_DPStepWatchdog | bool | None" = None


def _get_watchdog() -> "_DPStepWatchdog | None":
    global _watchdog
    if _watchdog is None:
        threshold = _read_threshold_seconds()
        _watchdog = _DPStepWatchdog(threshold) if threshold is not None else False
    return _watchdog or None


def heartbeat() -> None:
    """Record per-step forward progress (no-op unless the watchdog is enabled).

    Call this at the tightest per-step point that every DP rank passes each
    step, so a rank that stalls before the DP coordination all_reduce is
    detected and its stack is captured.
    """
    wd = _get_watchdog()
    if wd is not None:
        wd.beat()


def mark_idle() -> None:
    """Mark the worker as between steps so the watchdog does not false-fire."""
    wd = _get_watchdog()
    if wd is not None:
        wd.idle()
