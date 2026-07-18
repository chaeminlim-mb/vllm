#!/usr/bin/env bash
# =============================================================================
# Canonical in-house CX7 image build — every input pinned to an immutable ref.
#
# WHY THIS EXISTS
#   The images the runs used (vllm-mori-pd:inhouse-cx7-barrier and its base
#   vllm-rocm-base:inhouse) are frozen 2026-07-08 artifacts whose vLLM source
#   snapshot had NO git identity and whose AITER was pulled from a moving branch
#   (chaeminlim-mb/aiter@main == the broken 0.1.1.dev1941). This script rebuilds
#   the whole stack from KNOWN, PINNED inputs so every byte is accounted for.
#
# HOW THE PINS WERE ESTABLISHED (ground truth, triangulated)
#   - `docker history vllm-rocm-base:inhouse`  -> the effective build-args baked
#     into the working base (torch/triton/FA/MoRI commits, NIC, arch).
#   - Jaeyoun HANDOFF_golden_cx7_2026-07-10.md  -> the stage-2 vLLM build command
#     (NIC_BACKEND=none for CX7) + confirmation the golden/inhouse recipes are the
#     same, only the vLLM src differs.
#   - Jaeyoun build/Dockerfile.aiter-swap + run_aiter_swap_builds.sh  -> the
#     VALIDATED AITER is the ROCm/aiter v0.1.14 RELEASE (29/30 gsm8k MTP1), NOT
#     the base's source-built chaeminlim-mb/aiter@main.
#   NOTE: our tree's docker/Dockerfile.rocm_base is BYTE-IDENTICAL to Jaeyoun's,
#   so "canonical" = the same Dockerfile + correct pinned --build-args (no fork).
#
# PINNED INPUTS
#   ROCm base : rocm/dev-ubuntu-22.04:7.2.3-complete
#   PyTorch   : ROCm/pytorch              @ d0c8b1f3      (Dockerfile default; pinned here explicitly)
#   Triton    : ROCm/triton               @ 0f380657
#   FlashAttn : Dao-AILab/flash-attention @ 0e60e394
#   vision/audio: pytorch/vision v0.24.1 / pytorch/audio v2.9.0
#   MoRI      : chaeminlim-mb/mori         @ 6ad812c4     (in-house fork; override of default ROCm/mori@v1.1.0)
#   AITER     : ROCm/aiter                 @ v0.1.14 tag  (override of default v0.1.16.post2; see AITER note)
#   vLLM      : THIS source tree + the 2-part scheduler fix (git SHA printed at runtime)
#   NIC       : NIC_BACKEND=none  (CX7 / ConnectX)
#   Arch      : gfx942 (torch); gfx942;gfx950 (aiter/mori, fixed as ENV in the Dockerfile)
#
# AITER note (source vs wheel)
#   This script SOURCE-BUILDS AITER from the v0.1.14 *tag* (--build-arg AITER_BRANCH=v0.1.14),
#   which is the "fully from source" path. Jaeyoun VALIDATED the prebuilt release
#   *wheel* of the same version. They are the same release; if you want the exact
#   validated binary (and to skip AITER's long kernel compile), set USE_AITER_WHEEL=1
#   and stage 2 will pip-install the pinned wheel over the base instead.
#
# PREREQUISITES (the usual failure points — check these first)
#   - Build node must be able to PULL rocm/dev-ubuntu-22.04:7.2.3-complete.
#     It is NOT cached locally, and the nfs1f:5000 mirror was unreachable from ai2.
#     Build on the node that has registry access (ai1 = mi300x-01 = 10.1.1.181).
#   - chaeminlim-mb/mori clone auth: if that repo is private, export GITHUB_TOKEN.
#   - vllm-mori-pd:milestone4-local present locally (InferenceX donor, stage 3).
#   - Disk + TIME: stage 1 recompiles PyTorch/Triton/FA(/AITER) FROM SOURCE = hours.
#
# USAGE
#   ./docker/build-inhouse-cx7-canonical.sh [all|base|vllm|infx]   (default: all)
#   Run stage 1 ("base") once; iterate on your vLLM fix with "vllm" + "infx".
# =============================================================================
set -euo pipefail

STAGE="${1:-all}"
VLLM_SRC="${VLLM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
USE_AITER_WHEEL="${USE_AITER_WHEEL:-0}"

# Immutable pins
ROCM_BASE="rocm/dev-ubuntu-22.04:7.2.3-complete"
PYTORCH_BRANCH="d0c8b1f3"
TRITON_BRANCH="0f380657"
FA_BRANCH="0e60e394"
MORI_BRANCH="6ad812c4"
AITER_TAG="v0.1.14"
AITER_WHEEL_URL="https://github.com/ROCm/aiter/releases/download/v0.1.14/amd_aiter-0.1.14+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

# chaeminlim-mb/mori clone URL (inject token only if provided)
MORI_URL="https://github.com/chaeminlim-mb/mori.git"
[ -n "$GITHUB_TOKEN" ] && MORI_URL="https://${GITHUB_TOKEN}@github.com/chaeminlim-mb/mori.git"

# Output tags
BASE_TAG="vllm-rocm-base:inhouse-pinned"
VLLM_TAG="vllm-mori-pd:inhouse-cx7-schedfix"
FINAL_TAG="vllm-mori-pd:inhouse-cx7-schedfix-infx"

cd "$VLLM_SRC"
VLLM_REF="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
echo "=== canonical build from vLLM src: ${VLLM_SRC} @ ${VLLM_REF} (branch $(git branch --show-current 2>/dev/null || echo ?)) ==="

# Pin an image against `docker system prune` (daily cron @23:55 + LLMBoost) by
# holding a tiny keepalive container that references it. `sleep infinity`, no GPU,
# --restart unless-stopped so it survives daemon restarts. Re-run to refresh.
pin_image() {
  local img="$1" name
  name="pin-$(echo "$img" | tr '/:' '--')"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --restart unless-stopped --entrypoint sleep "$img" infinity >/dev/null \
    && echo "PINNED ${img}  (keepalive: ${name})" \
    || echo "WARN: failed to pin ${img}" >&2
}

build_base() {
  echo "### STAGE 1/3: canonical base ${BASE_TAG} (from-source, ~hours) ###"
  DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm_base \
    --build-arg BASE_IMAGE="${ROCM_BASE}" \
    --build-arg PYTORCH_ROCM_ARCH=gfx942 \
    --build-arg PYTORCH_REPO=https://github.com/ROCm/pytorch.git \
    --build-arg PYTORCH_BRANCH="${PYTORCH_BRANCH}" \
    --build-arg PYTORCH_VISION_BRANCH=v0.24.1 \
    --build-arg PYTORCH_AUDIO_BRANCH=v2.9.0 \
    --build-arg TRITON_REPO=https://github.com/ROCm/triton.git \
    --build-arg TRITON_BRANCH="${TRITON_BRANCH}" \
    --build-arg FA_REPO=https://github.com/Dao-AILab/flash-attention.git \
    --build-arg FA_BRANCH="${FA_BRANCH}" \
    --build-arg MORI_REPO="${MORI_URL}" \
    --build-arg MORI_BRANCH="${MORI_BRANCH}" \
    --build-arg AITER_REPO=https://github.com/ROCm/aiter.git \
    --build-arg AITER_BRANCH="${AITER_TAG}" \
    -t "${BASE_TAG}" .
  pin_image "${BASE_TAG}"
}

build_vllm() {
  echo "### STAGE 2/3: our vLLM (+ scheduler fix) on ${BASE_TAG} -> ${VLLM_TAG} ###"
  DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm \
    --build-arg BASE_IMAGE="${BASE_TAG}" \
    --build-arg ARG_PYTORCH_ROCM_ARCH=gfx942 \
    --build-arg NIC_BACKEND=none \
    --target final -t "${VLLM_TAG}" .

  if [ "${USE_AITER_WHEEL}" = "1" ]; then
    echo "### (USE_AITER_WHEEL=1) pinning EXACT validated AITER wheel ${AITER_TAG} over ${VLLM_TAG} ###"
    local tdf; tdf="$(mktemp)"
    printf 'FROM %s\nRUN python3 -m pip uninstall -y amd-aiter aiter 2>/dev/null || true && \\\n    python3 -m pip install --no-cache-dir "%s" && \\\n    python3 -c "import importlib.metadata as m; print(\x27amd-aiter ->\x27, m.version(\x27amd-aiter\x27))"\n' "${VLLM_TAG}" "${AITER_WHEEL_URL}" > "${tdf}"
    DOCKER_BUILDKIT=1 docker build -f "${tdf}" -t "${VLLM_TAG}" .
    rm -f "${tdf}"
  fi
  pin_image "${VLLM_TAG}"
}

build_infx() {
  echo "### STAGE 3/3: graft InferenceX -> ${FINAL_TAG} ###"
  if ! docker image inspect vllm-mori-pd:milestone4-local >/dev/null 2>&1; then
    echo "ERROR: vllm-mori-pd:milestone4-local not present (InferenceX donor). Load it first." >&2
    exit 1
  fi
  pin_image vllm-mori-pd:milestone4-local   # donor must survive the prune too
  local tdf; tdf="$(mktemp)"
  printf 'FROM %s\nCOPY --from=vllm-mori-pd:milestone4-local /app/InferenceX /app/InferenceX\n' "${VLLM_TAG}" > "${tdf}"
  DOCKER_BUILDKIT=1 docker build -f "${tdf}" -t "${FINAL_TAG}" .
  rm -f "${tdf}"
  pin_image "${FINAL_TAG}"
}

case "${STAGE}" in
  base) build_base ;;
  vllm) build_vllm ;;
  infx) build_infx ;;
  all)  build_base; build_vllm; build_infx ;;
  *)    echo "usage: $0 [all|base|vllm|infx]" >&2; exit 2 ;;
esac

echo "=== DONE (${STAGE}). Run image: ${FINAL_TAG} ==="
