# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force AITER rms_norm JIT modules to build during warmup, off the
per-step serving path.

WEDGE #2: the AITER rms_norm kernels are JIT-built by AITER lazily on first
*execution* inside the compiled inductor graph (``aiter_ops._rms_norm_impl`` ->
``aiter.rms_norm`` and the fused-add path -> ``aiter.rmsnorm2d_fwd_with_add``).
On a node whose AITER ``.so`` cache is cold, the first SERVING touch serializes
the same-node ranks on AITER's local ``file_baton`` while peer ranks block in
the per-step DP all_reduce -> gloo timeout -> instance crash.

AITER's build is keyed only by the module name (``aiter/jit/core.py``
``build_module(md_name, ...)``), so one idempotent first-touch per rank during
the v4-gated warmup phase is sufficient and serving-safe: it builds the ``.so``
if absent and is a cheap no-op if already present. Calling the identical
top-level ``aiter`` entry points used by ``vllm/kernels/aiter_ops.py`` guarantees
the same ``build_module`` calls serving would otherwise trigger:

* ``aiter.rms_norm``               -> builds ``module_rmsnorm``
* ``aiter.rmsnorm2d_fwd_with_add`` -> (hidden <= 8192) builds
  ``module_rmsnorm_quant``

The build itself is collective-free (the file_baton has no distributed calls),
so serializing it here cannot deadlock a serving DP all_reduce because none is
in flight during ``kernel_warmup``.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def aiter_rmsnorm_warmup(vllm_config) -> None:
    """Pre-build the AITER rms_norm JIT modules during the gated warmup phase.

    No-op unless: data_parallel_size > 1 (single-DP has no bracketing per-step
    DP all_reduce to deadlock), ROCm platform, AITER + AITER rmsnorm enabled,
    and ``aiter`` importable.
    """
    parallel_config = vllm_config.parallel_config
    # No-op for single-DP (no bracketing per-step DP all_reduce to deadlock).
    if parallel_config.data_parallel_size <= 1:
        return

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return

    import vllm.envs as envs

    if not (envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_RMSNORM):
        return

    try:
        from aiter import rms_norm, rmsnorm2d_fwd_with_add
    except Exception as e:
        logger.debug("AITER rms_norm warmup skipped (import failed): %s", e)
        return

    device = current_platform.device_type
    # Hidden <= 8192 matches every DeepSeek-R1 norm width (7168 / 1536 / 512).
    # AITER keys the build by module name only, so any small <=8192 16-bit
    # shape triggers the exact same .so build serving needs. The predicate
    # rms_no_var_16bit_only requires a 16-bit dtype, so touch both to remove
    # any doubt about eligibility; the second is a cheap cache hit.
    eps = 1e-6
    hidden = 7168
    try:
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.empty((1, hidden), dtype=dtype, device=device)
            w = torch.empty((hidden,), dtype=dtype, device=device)
            # Unfused: mirrors aiter_ops._rms_norm_impl -> aiter.rms_norm
            # (builds module_rmsnorm).
            rms_norm(x, w, eps)
            # Fused-add: mirrors
            # aiter_ops._rocm_aiter_rmsnorm2d_fwd_with_add_impl ->
            # aiter.rmsnorm2d_fwd_with_add (builds module_rmsnorm_quant for
            # hidden <= 8192). Argument order matches the serving call exactly.
            out = torch.empty_like(x)
            res_in = torch.empty_like(x)
            res_out = torch.empty_like(x)
            rmsnorm2d_fwd_with_add(out, x, res_in, res_out, w, eps)
        torch.cuda.synchronize()
    except Exception as e:
        # Never block warmup on this best-effort pre-build; if it fails the
        # behavior simply reverts to the prior (lazy) build path.
        logger.warning("AITER rms_norm warmup did not complete: %s", e)
        return

    logger.info("AITER rms_norm JIT modules pre-built during warmup.")
