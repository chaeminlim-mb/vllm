# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder


@pytest.fixture(autouse=True)
def _reset_mtp_split_env(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_AITER_MLA_MTP_DECODE_SPLIT", raising=False)
    yield
    envs.disable_envs_cache()


def _mtp_config(*, tp_size: int, num_speculative_tokens: int = 3):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="deepseek_mtp",
            num_speculative_tokens=num_speculative_tokens,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )


@pytest.mark.parametrize("tp_size", [1, 8])
def test_rocm_aiter_mtp_decode_uses_native_qlen_by_default(tp_size):
    config = _mtp_config(tp_size=tp_size)

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 4
    assert not AiterMLAMetadataBuilder._split_uniform_mtp_decode(config)
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize("tp_size", [1, 8])
def test_rocm_aiter_mtp_decode_split_fallback_is_opt_in(monkeypatch, tp_size):
    monkeypatch.setenv("VLLM_AITER_MLA_MTP_DECODE_SPLIT", "1")
    config = _mtp_config(tp_size=tp_size)

    assert AiterMLAMetadataBuilder._split_uniform_mtp_decode(config)
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize("tp_size", [1, 8])
def test_rocm_aiter_mtp_decode_splits_unsupported_native_qlen(tp_size):
    config = _mtp_config(tp_size=tp_size, num_speculative_tokens=4)

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 5
    assert AiterMLAMetadataBuilder._split_uniform_mtp_decode(config)
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_rocm_aiter_mla_without_mtp_keeps_single_only_decode():
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) is None
    assert not AiterMLAMetadataBuilder._split_uniform_mtp_decode(config)
    assert not AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
