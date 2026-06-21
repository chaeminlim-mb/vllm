# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import vllm.envs as envs

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder


def _mtp_config(
    *,
    tp_size: int,
    method: str = "deepseek_mtp",
    num_speculative_tokens: int | None = 3,
    num_spec_tokens: int | None = None,
    kv_connector: str | None = None,
    kv_role: str | None = None,
):
    kv_transfer_config = None
    if kv_connector is not None:
        kv_transfer_config = SimpleNamespace(
            kv_connector=kv_connector,
            kv_role=kv_role,
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    speculative_fields = {"method": method}
    if num_speculative_tokens is not None:
        speculative_fields["num_speculative_tokens"] = num_speculative_tokens
    if num_spec_tokens is not None:
        speculative_fields["num_spec_tokens"] = num_spec_tokens
    return SimpleNamespace(
        speculative_config=SimpleNamespace(**speculative_fields),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        kv_transfer_config=kv_transfer_config,
    )


@pytest.mark.parametrize("tp_size", [1, 8])
def test_rocm_aiter_mtp_decode_uses_native_qlen_by_default(tp_size):
    config = _mtp_config(tp_size=tp_size)

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 4
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert not AiterMLAMetadataBuilder._should_split_mtp_decode(config)


def test_rocm_aiter_mtp_decode_accepts_normalized_method_and_legacy_token_field():
    config = _mtp_config(
        tp_size=8,
        method="mtp",
        num_speculative_tokens=None,
        num_spec_tokens=3,
    )

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 4


@pytest.mark.parametrize(
    ("kv_role", "expected_split"),
    [
        ("kv_consumer", True),
        ("kv_both", True),
        ("kv_producer", False),
    ],
)
def test_rocm_aiter_mtp_decode_splits_moriio_consumers_by_default(
    kv_role,
    expected_split,
):
    config = _mtp_config(
        tp_size=8,
        kv_connector="MoRIIOConnector",
        kv_role=kv_role,
    )

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 4
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert AiterMLAMetadataBuilder._should_split_mtp_decode(config) is expected_split


def test_rocm_aiter_mtp1_decode_splits_moriio_consumer():
    config = _mtp_config(
        tp_size=8,
        num_speculative_tokens=1,
        kv_connector="MoRIIOConnector",
        kv_role="kv_consumer",
    )

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 2
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert AiterMLAMetadataBuilder._should_split_mtp_decode(config)


@pytest.mark.parametrize("tp_size", [1, 8])
def test_rocm_aiter_mtp_decode_splits_qlen_above_native_cap(tp_size):
    config = _mtp_config(tp_size=tp_size, num_speculative_tokens=4)

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 5
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert AiterMLAMetadataBuilder._should_split_mtp_decode(config)


def test_rocm_aiter_mtp_decode_env_can_force_split(monkeypatch):
    config = _mtp_config(tp_size=8)

    assert not AiterMLAMetadataBuilder._should_split_mtp_decode(config)
    monkeypatch.setenv("VLLM_AITER_MLA_MTP_DECODE_SPLIT", "1")
    envs.disable_envs_cache()
    try:
        assert AiterMLAMetadataBuilder._should_split_mtp_decode(config)
    finally:
        monkeypatch.delenv("VLLM_AITER_MLA_MTP_DECODE_SPLIT")
        envs.disable_envs_cache()


@pytest.mark.parametrize(
    ("tp_size", "expected"),
    [
        (1, True),
        (8, False),
    ],
)
def test_rocm_aiter_mtp_single_token_support_remains_tp1_only(tp_size, expected):
    config = _mtp_config(tp_size=tp_size, num_speculative_tokens=0)

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) == 1
    assert AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config) is expected


def test_rocm_aiter_mla_without_mtp_keeps_single_only_decode():
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    assert AiterMLAMetadataBuilder._mtp_decode_query_len(config) is None
    assert not AiterMLAMetadataBuilder._allow_uniform_mtp_decode(config)


def test_rocm_aiter_mtp_decode_detects_full_cg_padding_rows():
    qo_len = torch.tensor([4, 4, 0, 0], dtype=torch.int32)

    assert AiterMLAMetadataBuilder._needs_uniform_mtp_padding(
        qo_len, max_qo_len=4, num_decode_tokens=16
    )


def test_rocm_aiter_mtp_decode_detects_all_padded_full_cg_rows():
    qo_len = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

    assert (
        AiterMLAMetadataBuilder._uniform_padded_mtp_qo_len(
            qo_len, max_qo_len=0, num_decode_tokens=16
        )
        == 4
    )


@pytest.mark.parametrize(
    ("qo_len", "max_qo_len", "num_decode_tokens"),
    [
        ([4, 4], 4, 8),
        ([4, 3, 0], 4, 12),
        ([1, 1, 0], 1, 3),
        ([4, 4, 0, 0], 4, 8),
        ([4, 4, 0, 0], 4, 20),
        ([4, 0, 4, 0], 4, 16),
    ],
)
def test_rocm_aiter_mtp_decode_padding_rejects_non_full_cg_cases(
    qo_len, max_qo_len, num_decode_tokens
):
    assert not AiterMLAMetadataBuilder._needs_uniform_mtp_padding(
        torch.tensor(qo_len, dtype=torch.int32),
        max_qo_len=max_qo_len,
        num_decode_tokens=num_decode_tokens,
    )
