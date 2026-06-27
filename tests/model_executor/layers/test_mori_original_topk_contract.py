# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk


class _PrepareFinalizeUsesOriginalIds:
    def __init__(self):
        self.finalize_topk_ids = None
        self.finalize_topk_weights = None

    def finalize_uses_original_topk_ids(self):
        return True

    def supports_async(self):
        return False

    def prepare(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        global_num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant,
    ):
        topk_ids.add_(100)
        topk_weights.add_(100)
        return hidden_states, None, None, topk_ids + 1, topk_weights + 1

    def finalize(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
        self.finalize_topk_ids = topk_ids.clone()
        self.finalize_topk_weights = topk_weights.clone()
        output.copy_(fused_expert_output)


class _FusedExpertsStub:
    def __init__(self):
        self.moe_config = SimpleNamespace(moe_parallel_config=None)
        self.quant_config = object()
        self.expects_unquantized_inputs = False

    def finalize_weight_and_reduce_impl(self):
        return object()


def test_modular_finalize_uses_original_ids_but_prepared_weights(monkeypatch):
    prepare_finalize = _PrepareFinalizeUsesOriginalIds()
    kernel = mk.FusedMoEKernelModularImpl(prepare_finalize, _FusedExpertsStub())
    monkeypatch.setattr(
        kernel,
        "_fused_experts",
        lambda **kwargs: torch.full_like(kwargs["a1q"], 3),
    )

    hidden_states = torch.zeros((2, 2), dtype=torch.float32)
    w1 = torch.empty((1, 1, 1), dtype=torch.float32)
    w2 = torch.empty((1, 1, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[5], [6]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.float32)

    output = kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        global_num_experts=1,
    )

    torch.testing.assert_close(output, torch.full_like(hidden_states, 3))
    torch.testing.assert_close(
        prepare_finalize.finalize_topk_ids,
        torch.tensor([[5], [6]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        prepare_finalize.finalize_topk_weights,
        torch.tensor([[101.25], [101.75]], dtype=torch.float32),
    )
