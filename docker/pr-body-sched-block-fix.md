## Summary

Under MTP (spec decode) + P/D disaggregation with a **synchronous** KV-connector
read (MoRIIO READ mode, `load_kv_async=False`), the decode over-allocates KV
blocks on a request's initial external-KV-load step and then trips the
connector's downstream check:

```python
assert len(local_block_ids) <= len(remote_block_ids)
```

The prefill transferred N prompt blocks; the decode reserved N+1. On that first
step the scheduler still reserves speculative capacity even though the remote
produced only the prompt. Two sources over-allocate:

* the lookahead reservation (`num_lookahead_tokens`), and
* `pad_spec_decode`, which pads `num_new_tokens` to `1 + num_spec_tokens` to keep
  a uniform cudagraph shape.

Either can push the decode's block count one past what the prefill sent. The
existing lookahead guard only zeroed lookahead for the **async** path
(`load_kv_async=True`); the synchronous READ path was never covered.

## Fix

Gate **both** the lookahead reservation and `pad_spec_decode` on
`num_external_computed_tokens > 0`, so the initial external-load step allocates
exactly the prompt blocks (`== remote block count`). Both reservations resume on
the next running step (`num_external == 0`), so there is no correctness or
spec-decode-throughput change beyond that one step — this mirrors what the async
path already does.

The same assert reproduces on our golden branch, so this is a **latent P/D +
spec-decode bug, not a regression**.

## Why this is not duplicating an existing PR

No open PR touches this code path. This is split out of #48534 (a MoRIIO
connector-only change) to keep that PR's approved review intact — the two changes
are independent (one is `vllm/v1/core/sched/scheduler.py`, the other is the KV
connector). Duplicate-work search (`gh pr list --search "scheduler spec decode
block"`, `"num_lookahead_tokens external computed"`) surfaced nothing addressing
the external-KV-load block-count mismatch; the nearest (#45280, role-aware
spec-decode optimizations) is a perf/role refactor that does not touch this guard.

## Reproduction

TP8 1P1D, DeepSeek-R1, MTP1 (`num_speculative_tokens=1`), MoRIIO READ mode,
prompt=959, `block_size=64`:

```
MORIIO_BLOCK_MISMATCH ... num_external_tokens=958 num_prompt_tokens=959
  num_spec_tokens=0 num_preemptions=0 local_count=16 remote_count=15
AssertionError: assert len(local_block_ids) <= len(remote_block_ids)
```

Prefill transferred 15 prompt blocks; decode reserved 16 → assert. Fails
deterministically on every request's first external-load step.

## Test commands and results

End-to-end 1P1D TP8:TP8 MTP1 P/D disaggregation (MoRIIO READ mode), gsm8k via
`lm_eval`:

```bash
# prefill (node A) and decode (node B), DeepSeek-R1, MTP1, READ mode, TP8 each
# behind the MoRIIO toy proxy; full harness in docker/run-schedfix-test.sh
lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 \
        --model_args model=...,base_url=http://<proxy>/v1/completions,...
```

Result (512 samples, before fix: 0 completions — crash on first decode step):

| metric | value |
| --- | --- |
| gsm8k flexible-extract | **0.9473** |
| gsm8k strict-match | **0.8496** |
| completions | **512 / 512** |
| block-mismatch asserts | **0** |
| decode container exit | **0** |

## Model evaluation

The change affects scheduling/serving under spec decode. gsm8k accuracy above is
in-line with DeepSeek-R1 MTP1 expectations, and the run completed cleanly with
zero asserts and a clean decode shutdown, versus a hard crash on the first decode
step without the fix.

## AI assistance

AI assistance (Claude) was used to help root-cause and draft this change. The
submitter reviewed every changed line, ran the reproduction and the end-to-end
gsm8k validation above, and understands and defends the change end-to-end.
