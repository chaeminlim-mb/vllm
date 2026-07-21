#!/usr/bin/env bash
# =============================================================================
# One-shot MoRIIO 1P1D TP8:TP8 MTP1 (READ) validation for the schedfix image.
#
#   >>> LAUNCH FROM AI2 (10.1.1.182). <<<
#
#   prefill -> AI2 (local)             : port 8100, kv_producer
#   decode  -> AI3 (10.1.1.183, ssh)   : port 8201, kv_consumer
#   router  -> AI2 (local)             : port 10001, started after both are healthy
#   gsm8k   -> AI2 (local, lm_eval)    : via the router (skip with SKIP_EVAL=1)
#
# Prereqs: image loaded + pinned on BOTH AI2 and AI3; passwordless ssh AI2->AI3;
#          model at /models/models/DeepSeek-R1 on both nodes.
# Re-runnable: tears down mori-prefill / mori-decode / mori-proxy first.
# =============================================================================
set -uo pipefail

IMG="${IMG:-vllm-mori-pd:inhouse-cx7-schedfix-infx}"
DECODE_HOST="${DECODE_HOST:-10.1.1.183}"
SKIP_EVAL="${SKIP_EVAL:-0}"
PREFILL_HEALTH="http://10.1.1.182:8100/health"
DECODE_HEALTH="http://10.1.1.183:8201/health"
ROUTER_MODELS="http://10.1.1.182:10001/v1/models"
LOG="${LOG:-/home/edwin.lim/schedfix_gsm8k_$(date +%m%d_%H%M).log}"

die() { echo "ERROR: $*" >&2; exit 1; }
wait_ok() { # url name max_tries(10s each)
  local url="$1" name="$2" n="${3:-150}" i
  echo "waiting for ${name} (${url}) ..."
  for ((i=1;i<=n;i++)); do
    curl -sf -o /dev/null "$url" && { echo "  ${name} READY"; return 0; }
    sleep 10
  done
  die "timeout waiting for ${name}"
}

# ---------- preflight ----------
docker image inspect "$IMG" >/dev/null 2>&1 || die "$IMG not present on AI2 (load+pin it first)"
ssh "$DECODE_HOST" "docker image inspect '$IMG' >/dev/null 2>&1" || die "$IMG not present on AI3 ($DECODE_HOST)"

# ---------- clean prior run ----------
docker rm -f mori-prefill mori-proxy >/dev/null 2>&1 || true
ssh "$DECODE_HOST" 'docker rm -f mori-decode >/dev/null 2>&1 || true'

# ---------- MoRIIO toy proxy on AI2 FIRST (owns HTTP :10001 + ping/discovery :36367) ----------
echo "### starting TOY PROXY on AI2 (first) ###"
docker run -d --name mori-proxy --entrypoint "" --network host \
  "$IMG" \
  python3 /app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py \
    --port 10001 \
  || die "toy proxy container failed to start"
sleep 8   # let it bind :10001 / :36367 before the engines register

# ---------- prefill on AI2 (local) ----------
echo "### starting PREFILL on AI2 ###"
docker run -d --name mori-prefill --network host --device /dev/kfd --device /dev/dri \
  $(for d in /dev/infiniband/*; do [ -e "$d" ] && printf ' --device %s' "$d"; done) \
  --group-add video --ipc host --privileged --cap-add SYS_ADMIN --cap-add IPC_LOCK --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --shm-size 192G --ulimit nofile=1048576:1048576 \
  -v /models/models:/models/models:ro -v /dev/shm:/dev/shm \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_LINEAR=True -e VLLM_ROCM_USE_AITER_MLA=True \
  -e VLLM_ROCM_USE_AITER_MOE=True -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
  -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_MORIIO_CONNECTOR_READ_MODE=true -e VLLM_LOGGING_LEVEL=INFO \
  -e AITER_ENABLE_VSKIP=0 -e AITER_ONLINE_TUNE=0 -e AITER_BYPASS_TUNE_CONFIG=1 -e AITER_USE_CK_MOE_SORTING=0 \
  -e VLLM_AITER_MLA_MTP_DECODE_SPLIT=0 \
  -e MORI_EP_LAUNCH_CONFIG_MODE=AUTO -e MORI_SHMEM_HEAP_SIZE=24G -e MORI_SHMEM_MODE=ISOLATION -e MORI_APP_LOG_LEVEL=INFO \
  -e MORI_IO_QP_MAX_SEND_WR=8192 -e MORI_IO_QP_MAX_CQE=16384 -e MORI_IO_QP_MAX_SGE=4 \
  -e MORI_RDMA_TC=104 -e MORI_IB_PATH_MTU=1024 -e NCCL_IB_DISABLE=0 \
  -e VLLM_MORIIO_QP_PER_TRANSFER=4 -e VLLM_MORIIO_POST_BATCH_SIZE=2 -e VLLM_MORIIO_NUM_WORKERS=1 \
  -e VLLM_MORIIO_HANDSHAKE_WORKERS=2 -e VLLM_MORIIO_HANDSHAKE_WAIT_TIMEOUT_S=120 \
  -e VLLM_MORIIO_MAX_INFLIGHT_GLOBAL=16 -e VLLM_MORIIO_MAX_INFLIGHT_PER_TRANSFER=4 -e VLLM_MORIIO_MAX_DISPATCH_LAYERS=4 \
  -e MORI_IO_LOG_LEVEL=ERROR -e MORI_GLOBAL_LOG_LEVEL=error -e MORI_PIN_PERRANK_CNIC=0 \
  -e MORI_IB_GID_INDEX=3 -e NCCL_IB_GID_INDEX=3 -e MORI_RDMA_DEVICES=rocep6s0f0,rocep35s0f0 \
  -e NCCL_IB_HCA==rocep6s0f0,=rocep35s0f0 \
  -e VLLM_HOST_IP=10.1.1.182 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.1.1.182 -e VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$IMG" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/models/DeepSeek-R1 --served-model-name DeepSeek-R1-0528 --host 0.0.0.0 --port 8100 \
    --tensor-parallel-size 8 --max-model-len 20480 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.80 \
    --dtype auto --trust-remote-code --distributed-executor-backend mp --kv-cache-dtype fp8_e4m3 \
    --block-size 64 --no-enable-chunked-prefill --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_rope_kvcache":true,"fuse_rope_kvcache_cat_mla":true,"enable_sp":true,"fuse_gemm_comms":true,"fuse_act_padding":true,"fuse_allreduce_rms":false}}' \
    --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}' \
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_connector_extra_config":{"proxy_ip":"10.1.1.182","proxy_port":36367,"proxy_ping_port":36367,"http_port":8100,"handshake_port":6301,"notify_port":61005,"read_mode":true,"qp_per_transfer":4,"post_batch_size":2,"num_workers":1,"handshake_timeout":120,"max_inflight_global":16,"max_inflight_per_transfer":4,"max_dispatch_layers":4}}' \
  || die "prefill container failed to start"

# ---------- decode on AI3 (remote; $(...) and JSON evaluate ON AI3) ----------
echo "### starting DECODE on AI3 (${DECODE_HOST}) ###"
ssh "$DECODE_HOST" 'bash -s' "$IMG" <<'REMOTE'
IMG="$1"
docker run -d --name mori-decode --network host --device /dev/kfd --device /dev/dri \
  $(for d in /dev/infiniband/*; do [ -e "$d" ] && printf ' --device %s' "$d"; done) \
  --group-add video --ipc host --privileged --cap-add SYS_ADMIN --cap-add IPC_LOCK --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --shm-size 192G --ulimit nofile=1048576:1048576 \
  -v /models/models:/models/models:ro -v /dev/shm:/dev/shm \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_LINEAR=True -e VLLM_ROCM_USE_AITER_MLA=True \
  -e VLLM_ROCM_USE_AITER_MOE=True -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
  -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_MORIIO_CONNECTOR_READ_MODE=true -e VLLM_LOGGING_LEVEL=INFO \
  -e AITER_ENABLE_VSKIP=0 -e AITER_ONLINE_TUNE=0 -e AITER_BYPASS_TUNE_CONFIG=1 -e AITER_USE_CK_MOE_SORTING=0 \
  -e VLLM_AITER_MLA_MTP_DECODE_SPLIT=0 \
  -e MORI_EP_LAUNCH_CONFIG_MODE=AUTO -e MORI_SHMEM_HEAP_SIZE=24G -e MORI_SHMEM_MODE=ISOLATION -e MORI_APP_LOG_LEVEL=INFO \
  -e MORI_IO_QP_MAX_SEND_WR=8192 -e MORI_IO_QP_MAX_CQE=16384 -e MORI_IO_QP_MAX_SGE=4 \
  -e MORI_RDMA_TC=104 -e MORI_IB_PATH_MTU=1024 -e NCCL_IB_DISABLE=0 \
  -e VLLM_MORIIO_QP_PER_TRANSFER=4 -e VLLM_MORIIO_POST_BATCH_SIZE=2 -e VLLM_MORIIO_NUM_WORKERS=1 \
  -e VLLM_MORIIO_HANDSHAKE_WORKERS=2 -e VLLM_MORIIO_HANDSHAKE_WAIT_TIMEOUT_S=120 \
  -e VLLM_MORIIO_MAX_INFLIGHT_GLOBAL=16 -e VLLM_MORIIO_MAX_INFLIGHT_PER_TRANSFER=4 -e VLLM_MORIIO_MAX_DISPATCH_LAYERS=4 \
  -e MORI_IO_LOG_LEVEL=ERROR -e MORI_GLOBAL_LOG_LEVEL=error -e MORI_PIN_PERRANK_CNIC=0 \
  -e MORI_IB_GID_INDEX=3 -e NCCL_IB_GID_INDEX=3 -e MORI_RDMA_DEVICES=rocep6s0f0,rocep35s0f0 \
  -e NCCL_IB_HCA==rocep6s0f0,=rocep35s0f0 \
  -e VLLM_HOST_IP=10.1.1.183 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.1.1.182 -e VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$IMG" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/models/DeepSeek-R1 --served-model-name DeepSeek-R1-0528 --host 0.0.0.0 --port 8201 \
    --tensor-parallel-size 8 --max-model-len 20480 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.80 \
    --dtype auto --trust-remote-code --distributed-executor-backend mp --kv-cache-dtype fp8_e4m3 \
    --block-size 64 --no-enable-chunked-prefill --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_rope_kvcache":true,"fuse_rope_kvcache_cat_mla":true,"enable_sp":true,"fuse_gemm_comms":true,"fuse_act_padding":true,"fuse_allreduce_rms":false}}' \
    --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}' \
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"proxy_ip":"10.1.1.182","proxy_port":36367,"proxy_ping_port":36367,"http_port":8201,"handshake_port":6301,"notify_port":61005,"read_mode":true,"qp_per_transfer":4,"post_batch_size":2,"num_workers":1,"handshake_timeout":120,"max_inflight_global":16,"max_inflight_per_transfer":4,"max_dispatch_layers":4}}'
REMOTE
[ $? -eq 0 ] || die "decode container failed to start on AI3"

# ---------- wait for both engines (router already up; they register as they come healthy) ----------
wait_ok "$PREFILL_HEALTH" "prefill(AI2:8100)"
wait_ok "$DECODE_HEALTH"  "decode(AI3:8201)"

# ---------- smoke test (proxy should now have both backends) ----------
echo "### smoke test ###"
curl -s http://10.1.1.182:10001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-R1-0528","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":50,"temperature":0}'
echo

if [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=1 -> servers+proxy up. Run gsm8k yourself. Logs: docker logs mori-prefill | mori-decode | mori-proxy"
  exit 0
fi

# ---------- gsm8k (the workload that crashed ~req 161 pre-fix) ----------
echo "### gsm8k (limit 512, c128) -> ${LOG} ###"
docker run --rm --network host "$IMG" bash -c '
pip install -q --break-system-packages "lm_eval[api]" 2>&1 | tail -1 || pip install -q "lm_eval[api]"
python3 - <<PY
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/lm_eval/models/api_models.py")
if p.exists():
    s = p.read_text()
    s = s.replace("outputs = await response.json()", "outputs = await response.json(content_type=None)")
    p.write_text(s); print("patched lm_eval content_type")
PY
python3 -m lm_eval --model local-chat-completions --apply_chat_template \
  --tasks gsm8k --limit 512 \
  --model_args "model=DeepSeek-R1-0528,base_url=http://10.1.1.182:10001/v1/chat/completions,api_key=EMPTY,num_concurrent=128,timeout=1800,tokenized_requests=False,max_length=20480" \
  --gen_kwargs "max_tokens=16384,temperature=0,top_p=1"
' 2>&1 | tee "$LOG"

# ---------- verdict: did the block-accounting assert fire on the decode? ----------
echo "### decode assert check (want 0) ###"
ssh "$DECODE_HOST" 'docker logs mori-decode 2>&1 | grep -cE "AssertionError|MORIIO_BLOCK_MISMATCH|local_block_ids <= remote_block_ids"' \
  | awk '{print "  decode assert hits:", $1}'
echo "eval log: ${LOG}"
