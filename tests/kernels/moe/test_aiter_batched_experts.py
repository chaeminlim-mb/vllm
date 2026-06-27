# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-level tests for the AITER ``BatchedExperts`` FP8 wrapper.

These tests guard reshape and oracle-selection contracts without invoking
AITER kernels. The wrapper test still runs through the real ``AiterExperts``
adapter and monkeypatches only the final AITER kernel call.
Covered contracts:

  * the wrapper advertises ``BatchedExperts`` activation format,
  * BatchedExperts prepare/finalize can provide already-quantized activations,
  * the wrapper flattens ``(E_local, M_e, K)`` activations and dispatched
    scales before delegating to the Standard-layout AITER experts,
  * ``BATCHED_AITER`` maps to ``AiterBatchedExpertsFp8``, and
  * the FP8 oracle routes the batched ROCm AITER path to that wrapper.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterBatchedExpertsFp8,
    AiterExperts,
)
from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    _get_priority_backends,
    backend_to_kernel_cls,
    select_fp8_moe_backend,
)

pytestmark = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="AITER BatchedExperts tests require supported ROCm AITER.",
)


def _make_moe_config(
    *,
    num_experts: int,
    hidden_dim: int,
    intermediate_size: int,
    max_num_tokens: int,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.float32,
        max_num_tokens=max_num_tokens,
    )


def _parallel_config(all2all_backend: str) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=2,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend=all2all_backend,
        enable_eplb=False,
    )


def test_aiter_batched_experts_fp8_activation_format():
    """The wrapper must advertise ``BatchedExperts`` format."""
    assert (
        AiterBatchedExpertsFp8.activation_format()
        == mk.FusedMoEActivationFormat.BatchedExperts
    )
    # And the sibling Standard variant must still be Standard.
    assert AiterExperts.activation_format() == mk.FusedMoEActivationFormat.Standard


def test_aiter_batched_experts_does_not_expect_unquantized_inputs():
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


@pytest.mark.parametrize(
    ("all2all_backend", "expected"),
    [
        ("deepep_low_latency", True),
        ("nixl_ep", True),
        ("flashinfer_nvlink_one_sided", False),
        ("flashinfer_nvlink_two_sided", False),
    ],
)
def test_aiter_batched_experts_uses_batched_activation_property(
    all2all_backend,
    expected,
):
    parallel_config = _parallel_config(all2all_backend)

    assert parallel_config.use_batched_experts_activation_format is expected
    assert AiterBatchedExpertsFp8._supports_parallel_config(parallel_config) is expected


@pytest.mark.parametrize(
    (
        "num_experts",
        "tokens_per_expert",
        "hidden_dim",
        "scale_dim",
        "expert_counts",
    ),
    [
        (1, 4, 8, None, [3]),
        (2, 3, 4, 2, [2, 1]),
        (3, 2, 5, 1, [2, 0, 1]),
    ],
)
def test_aiter_batched_experts_flattens_batched_layout_for_inner_aiter(
    monkeypatch,
    num_experts,
    tokens_per_expert,
    hidden_dim,
    scale_dim,
    expert_counts,
):
    """Check the batched-to-flat wrapper contract without invoking kernels."""
    hidden_states = torch.arange(
        num_experts * tokens_per_expert * hidden_dim, dtype=torch.float32
    ).reshape(num_experts, tokens_per_expert, hidden_dim)
    output = torch.empty_like(hidden_states)
    if scale_dim is None:
        a1q_scale = None
        expected_a1q_scale = None
    else:
        a1q_scale = torch.arange(
            num_experts * tokens_per_expert * scale_dim, dtype=torch.float32
        ).reshape(num_experts, tokens_per_expert, scale_dim)
        expected_a1q_scale = a1q_scale.reshape(
            num_experts * tokens_per_expert, scale_dim
        )
    a2_scale = torch.tensor([0.5])
    expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(expert_counts, "cpu")
    captured = {}

    def fake_rocm_aiter_fused_experts(**kwargs):
        captured.update(kwargs)
        routed_ids = kwargs["topk_ids"].to(dtype=kwargs["hidden_states"].dtype)
        return kwargs["hidden_states"] + routed_ids

    monkeypatch.setattr(
        rocm_aiter_moe,
        "rocm_aiter_fused_experts",
        fake_rocm_aiter_fused_experts,
    )
    wrapper = AiterBatchedExpertsFp8(
        _make_moe_config(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_size=1,
            max_num_tokens=tokens_per_expert,
        ),
        FUSED_MOE_UNQUANTIZED_CONFIG,
        max_num_tokens=tokens_per_expert,
        num_dispatchers=1,
    )

    wrapper.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(num_experts, 1, 1),
        w2=torch.empty(num_experts, 1, 1),
        topk_weights=torch.empty(1, 1),
        topk_ids=torch.empty(1, 1, dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=99,
        expert_map=torch.arange(num_experts),
        a1q_scale=a1q_scale,
        a2_scale=a2_scale,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=expert_tokens_meta,
        apply_router_weight_on_input=True,
    )

    expected_ids = (
        torch.arange(num_experts, dtype=torch.int32)
        .repeat_interleave(tokens_per_expert)
        .unsqueeze(-1)
    )

    assert captured["hidden_states"].shape == (
        num_experts * tokens_per_expert,
        hidden_dim,
    )
    assert torch.equal(
        captured["hidden_states"],
        hidden_states.reshape(num_experts * tokens_per_expert, hidden_dim),
    )
    assert torch.equal(captured["topk_ids"], expected_ids)
    assert torch.equal(
        captured["topk_weights"], torch.ones(num_experts * tokens_per_expert, 1)
    )
    assert captured["moe_config"].num_experts == num_experts
    assert captured["global_num_experts"] == num_experts
    assert captured["expert_map"] is None
    if expected_a1q_scale is None:
        assert captured["a1q_scale"] is None
    else:
        assert torch.equal(captured["a1q_scale"], expected_a1q_scale)
    assert captured["num_local_tokens"] is None
    assert captured["expert_tokens_meta"] is None
    assert captured["apply_router_weight_on_input"] is False

    expected_output = (
        hidden_states.reshape(num_experts * tokens_per_expert, hidden_dim)
        + expected_ids.float()
    )
    assert torch.equal(
        output,
        expected_output.reshape(num_experts, tokens_per_expert, hidden_dim),
    )


def test_oracle_registers_batched_aiter_backend():
    """``BATCHED_AITER`` must exist in ``Fp8MoeBackend`` and map to the
    wrapper class."""
    # Enum membership.
    assert Fp8MoeBackend.BATCHED_AITER.value == "BATCHED_AITER"
    # Dispatch table mapping.
    classes = backend_to_kernel_cls(Fp8MoeBackend.BATCHED_AITER)
    assert classes == [AiterBatchedExpertsFp8]


def test_select_fp8_moe_backend_routes_batched_aiter_env_to_wrapper(monkeypatch):
    """ROCm AITER env selection must pick the batched wrapper for DP/EP."""

    def is_set(name):
        return name in {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"}

    def is_supported(cls, config, weight_key, activation_key, activation_format):
        assert activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        return True, None

    config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=True,
            use_deepep_v2_kernels=False,
            ep_size=1,
        ),
    )
    monkeypatch.setattr(fp8_oracle.envs, "is_set", is_set)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_ROCM_USE_AITER", True)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_ROCM_USE_AITER_MOE", True)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)
    monkeypatch.setattr(
        AiterBatchedExpertsFp8,
        "is_supported_config",
        staticmethod(is_supported),
    )

    backend, experts_cls = select_fp8_moe_backend(config, None, None)

    assert backend == Fp8MoeBackend.BATCHED_AITER
    assert experts_cls is AiterBatchedExpertsFp8


def test_oracle_priority_order_places_batched_aiter_before_fallbacks():
    # Only these fields are needed for the ROCm/default priority order checked here.
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_deepep_v2_kernels=False,
            ep_size=1,
        )
    )
    backends = _get_priority_backends(moe_config, None, None)

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
