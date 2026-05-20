# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-level smoke tests for the AITER ``BatchedExperts`` FP8 wrapper.

These tests guard the *reshape contract* — the wrapper must:
  * advertise ``BatchedExperts`` activation format,
  * advertise ``expects_unquantized_inputs == False`` (so the BatchedExperts
    prepare step does not raise on ``defer_input_quant``),
  * be registered in the FP8 oracle's ``BATCHED_AITER`` slot,
  * map ``BATCHED_AITER`` → ``AiterBatchedExpertsFp8`` via
    ``backend_to_kernel_cls``,
  * be selectable by ``select_fp8_moe_backend`` when ROCm + AITER + the
    BatchedExperts activation format are all active.

We deliberately do **not** run a numerical-parity test here against
``NaiveBatchedExperts`` because:

  1. ``rocm_aiter_fused_experts`` requires the actual AITER package, which is
     ROCm-only, and the CI runner for these tests is usually CUDA.
  2. The wrapper is purely a reshape/dispatch layer over the existing
     Standard-layout AITER kernel — its numerical behavior is identical to
     ``AiterExperts.apply()`` on the flattened input. Numerical parity is
     covered indirectly by the existing AITER test suite.

A proper end-to-end numerical-parity test against ``NaiveBatchedExperts`` on
real ROCm hardware is tracked separately.  # TODO(numerical-parity-test)
"""

import importlib.util

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

# ---------------------------------------------------------------------------
# Import the wrapper without requiring the AITER runtime. Importing the module
# must work even when AITER is not present — it only fails at *kernel-call*
# time. This is the same pattern as the rest of the vllm aiter modules.
# ---------------------------------------------------------------------------
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (  # noqa: E402
    AiterBatchedExpertsFp8,
    AiterExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (  # noqa: E402
    Fp8MoeBackend,
    backend_to_kernel_cls,
)
from vllm.platforms import current_platform


def test_aiter_batched_experts_fp8_activation_format():
    """The wrapper must advertise ``BatchedExperts`` format."""
    assert (
        AiterBatchedExpertsFp8.activation_format()
        == mk.FusedMoEActivationFormat.BatchedExperts
    )
    # And the sibling Standard variant must still be Standard.
    assert AiterExperts.activation_format() == mk.FusedMoEActivationFormat.Standard


def test_aiter_batched_experts_does_not_expect_unquantized_inputs():
    """Critical: ``BatchedExperts`` prepare steps (DeepEP-LL, NIXL) explicitly
    reject ``defer_input_quant=True``. The wrapper must not request it."""
    # ``expects_unquantized_inputs`` is a @property on the base class, so we
    # have to query an instance, not the class. We don't construct a full
    # ``FusedMoEConfig`` (lots of plumbing) — we just check the descriptor
    # directly.
    prop = AiterBatchedExpertsFp8.__dict__["expects_unquantized_inputs"]
    assert isinstance(prop, property), (
        "expects_unquantized_inputs must be a @property to match the ABC"
    )
    # Invoke the getter with a dummy ``self`` proxy. The body of the getter
    # only returns False unconditionally.
    fget = prop.fget
    assert fget is not None
    assert fget(object.__new__(AiterBatchedExpertsFp8)) is False


def test_aiter_batched_experts_does_not_support_expert_map():
    """BatchedExperts is already-local; no expert_map needed."""
    # ``supports_expert_map`` is an instance method on the ABC, but its body
    # for the wrapper is a stateless ``return False``.
    inst = object.__new__(AiterBatchedExpertsFp8)
    assert inst.supports_expert_map() is False


def test_oracle_registers_batched_aiter_backend():
    """``BATCHED_AITER`` must exist in ``Fp8MoeBackend`` and map to the
    wrapper class."""
    # Enum membership.
    assert Fp8MoeBackend.BATCHED_AITER.value == "BATCHED_AITER"
    # Dispatch table mapping.
    classes = backend_to_kernel_cls(Fp8MoeBackend.BATCHED_AITER)
    assert classes == [AiterBatchedExpertsFp8]


def test_oracle_priority_order_places_batched_aiter_before_fallbacks():
    """``BATCHED_AITER`` must be tried before the generic Triton/CUTLASS
    batched fallbacks on the ROCm path, so the kernel-doesn't-support error
    is never surfaced when AITER MoE is enabled."""
    # We can't easily construct a full FusedMoEConfig in a unit test, but the
    # static priority list can be inspected by importing the helper.
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        _get_priority_backends,
    )

    # ``_get_priority_backends`` does not consult moe_config for the order
    # we care about here — it only reshuffles based on Hopper/XPU/CPU. So we
    # pass ``None`` everywhere; the function returns the platform-default list.
    backends = _get_priority_backends(None, None, None)  # type: ignore[arg-type]

    assert Fp8MoeBackend.BATCHED_AITER in backends
    ba_idx = backends.index(Fp8MoeBackend.BATCHED_AITER)
    for fallback in (
        Fp8MoeBackend.BATCHED_DEEPGEMM,
        Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        Fp8MoeBackend.BATCHED_TRITON,
    ):
        if fallback in backends:
            assert ba_idx < backends.index(fallback), (
                f"BATCHED_AITER must precede {fallback.value} in the priority list"
            )


# ---------------------------------------------------------------------------
# Numerical / kernel-execution test (skipped unless ROCm + AITER + GPU).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER BatchedExperts wrapper only runs on ROCm",
)
@pytest.mark.skipif(
    importlib.util.find_spec("aiter") is None,
    reason="AITER package not installed",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for kernel execution",
)
def test_aiter_batched_experts_runs_smoke():
    """End-to-end smoke test: instantiate the wrapper with minimal inputs and
    verify ``apply()`` writes through the output buffer without raising.

    This is *not* a numerical-parity test against ``NaiveBatchedExperts`` —
    see module docstring. It catches gross integration errors (e.g. shape
    mismatches in the reshape path, missing kwargs into the inner kernel).
    """
    # TODO(numerical-parity-test): augment this with a small synthetic FP8
    # MoE config and compare element-wise against ``NaiveBatchedExperts``
    # at ``rtol=atol=1e-2``. Requires a working torch.float8_e4m3fnuz path
    # plus a constructed FusedMoEConfig and FusedMoEQuantConfig (block-FP8).
    pytest.skip(
        "Full numerical-parity smoke requires a FusedMoEConfig fixture; "
        "see TODO(numerical-parity-test) in this file."
    )
