#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve the ultrafast build of Qwen3.8-Flash-Next on a Jetson AGX Thor.
#
# This is the upstream recipe's v16b launch.sh and serve.sh merged into one
# script, with the Thor deltas. It does three things:
#
#   1. loads config/ultrafast/env (a value already exported by the caller wins
#      over the file's, the way this repo's root serve.sh hands off to
#      scripts/serve-intel-ar.sh);
#   2. expands config/ultrafast/draft-vocab-ids-K65536.txt.gz into
#      $HOME/.cache/qwen38-ultrafast/, but mounts it only after the expanded
#      file is proven to hold exactly 65536 non-empty lines;
#   3. composes the docker run command (the speculative config is merged with
#      python3, since jq is not a dependency we want to add on the device).
#
#   scripts/serve-ultrafast.sh --print    # print the command + env, launch nothing
#   scripts/serve-ultrafast.sh            # expand the vocab, then launch
#
# The image tag is fixed (no :latest) on purpose, as upstream requires: the
# numbers in the README belong to one specific image.
set -euo pipefail

case "${1:-}" in
  --print) print=1 ;;
  --run|'') print=0 ;;
  *) echo "serve-ultrafast.sh: unknown argument '$1' (use --print or --run)" >&2; exit 64 ;;
esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

cd "$(dirname "$0")/.."
envfile='config/ultrafast/env'
vocab_gz='config/ultrafast/draft-vocab-ids-K65536.txt.gz'
[ -r "$envfile" ] && [ -r "$vocab_gz" ] || {
  echo "serve-ultrafast.sh: missing $envfile or $vocab_gz" >&2; exit 65;
}

# env file, but never overwriting what the caller already exported. Lines are
# eval'd, so "$HOME" inside a value expands here rather than at use time.
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  case "$line" in *=*) ;; *) continue ;; esac
  key="${line%%=*}"
  case "$key" in [A-Za-z_]*) ;; *) continue ;; esac
  case "$key" in *[!A-Za-z0-9_]*) continue ;; esac
  if [ -z "${!key+set}" ]; then
    eval "export $line"
  fi
done < "$envfile"

# --- defaults for a direct call (the env file supplies all of these) --------
NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx:ultrafast-thor-20261006}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32}"
TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.01}"
MTP="${MTP:-3}"
PREWARM="${PREWARM:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
EXTRA="${EXTRA:-}"
KV_BYTES="${KV_BYTES:-20g}"
EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

# CUDA-graph splitting: the PLE gather is a CPU op plus an H2D copy, so it must
# stay outside the graphs. Identical to scripts/serve-intel-ar.sh.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = 1 ] && AT_ARG=

# --- speculative config: {"method":"mtp",...} merged with SPEC_EXTRA --------
# python3 instead of jq: the device has python3 already, and adding a jq
# dependency to the box is not worth one dictionary merge.
SPEC=()
if [ "$MTP" != 0 ]; then
  SPEC_JSON='{"method":"mtp","num_speculative_tokens":'"$MTP"'}'
  if [ -n "${SPEC_EXTRA:-}" ]; then
    if ! SPEC_JSON="$(python3 -c '
import json, sys
a = json.loads(sys.argv[1])
b = json.loads(sys.argv[2])
if not isinstance(b, dict):
    raise SystemExit("SPEC_EXTRA must be a JSON object")
a.update(b)
print(json.dumps(a, separators=(",", ":")))
' "$SPEC_JSON" "$SPEC_EXTRA" 2>&1)"; then
      echo "serve-ultrafast.sh: SPEC_EXTRA is not a valid JSON object: $SPEC_EXTRA" >&2
      exit 2
    fi
  fi
  SPEC=(--speculative-config "$SPEC_JSON")
fi

PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-1}" = 1 ] && PC_ARG=--enable-prefix-caching

PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ] && [ "${PREFIX_CACHE:-0}" = 1 ]; then
  PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
           --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
fi

# --- v16b identity knobs ----------------------------------------------------
PLE_FAST_PATH="${PLE_FAST_PATH:-1}"
PLE_FAST_MAX_ROWS="${PLE_FAST_MAX_ROWS:-262144}"
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-${HOME}/.cache/qwen38-ultrafast/draft-vocab-ids-K65536.txt}"
DRAFTER_EXPERTS_FP8="${DRAFTER_EXPERTS_FP8:-1}"
LOW_LATENCY_GEMM="${LOW_LATENCY_GEMM:-1}"
LLG_PDL="${LLG_PDL:-0}"
VERIFY_TOPK_TRITON="${VERIFY_TOPK_TRITON:-1}"
KEEP_DRAFT_BLOCKS="${KEEP_DRAFT_BLOCKS:-1}"
# Thor: the two sm_110 workarounds, also accepting the root serve.sh's short
# spellings (QSA_EXACT_TOPK / GDN_DECODE_KERNEL) as aliases.
VLLM_QSA_EXACT_TOPK="${VLLM_QSA_EXACT_TOPK:-${QSA_EXACT_TOPK:-1}}"
VLLM_GDN_DECODE_KERNEL="${VLLM_GDN_DECODE_KERNEL:-${GDN_DECODE_KERNEL:-triton}}"

# One flat list of container env, so --print can print exactly what is set
# without a second copy of the list drifting away from the real one.
ENV_ARGS=(
  -e VLLM_PLE_MMAP=1
  -e "VLLM_PLE_MMAP_WORKERS=${WORKERS:-32}"
  -e "VLLM_PLE_MMAP_PREWARM=${PREWARM:-1}"
  -e "VLLM_PLE_MMAP_PREFETCH=${PLE_PREFETCH:-0}"
  -e "VLLM_PLE_MMAP_MADV_RANDOM=${PLE_MADV_RANDOM:-0}"
  -e VLLM_PLE_MMAP_DIR=/ple-table
  -e "VLLM_PLE_MMAP_FAST_PATH=${PLE_FAST_PATH}"
  -e "VLLM_PLE_MMAP_FAST_MAX_ROWS=${PLE_FAST_MAX_ROWS}"
  -e "VLLM_HIT_DEBUG=${HIT_DEBUG:-0}"
  -e "VLLM_STEP_PROFILE=${STEP_PROFILE:-0}"
  -e "VLLM_QSA_EXACT_TOPK=${VLLM_QSA_EXACT_TOPK}"
  -e "VLLM_GDN_DECODE_KERNEL=${VLLM_GDN_DECODE_KERNEL}"
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1
  -e "VLLM_FP8_HYBRID=${FP8_HYBRID:-1}"
  -e VLLM_USE_DEEP_GEMM=0
  -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
)
# Optional PLE knobs: omitted (never set to empty) when unset, as upstream does.
[ -n "${PLE_FAST_ROWS:-}" ] && ENV_ARGS+=(-e "VLLM_PLE_MMAP_FAST_ROWS=$PLE_FAST_ROWS")
[ -n "${PLE_CHUNK:-}" ]     && ENV_ARGS+=(-e "VLLM_PLE_MMAP_CHUNK=$PLE_CHUNK")
[ -n "${PLE_STATS_SEC:-}" ] && ENV_ARGS+=(-e "VLLM_PLE_MMAP_STATS_SEC=$PLE_STATS_SEC")
# The v16b patch knobs: L5a, R8, R3 and the drafter-experts fp8 switch.
ENV_ARGS+=(-e "VLLM_VERIFY_TOPK_TRITON=$VERIFY_TOPK_TRITON")
ENV_ARGS+=(-e "VLLM_KEEP_DRAFT_BLOCKS=$KEEP_DRAFT_BLOCKS")
[ -n "$LOW_LATENCY_GEMM" ] && ENV_ARGS+=(-e "QWEN38NEXT_LOW_LATENCY_GEMM=$LOW_LATENCY_GEMM")
[ -n "$LLG_PDL" ]          && ENV_ARGS+=(-e "QWEN38NEXT_LLG_PDL=$LLG_PDL")
[ -n "$DRAFTER_EXPERTS_FP8" ] && ENV_ARGS+=(-e "VLLM_DRAFTER_EXPERTS_FP8=$DRAFTER_EXPERTS_FP8")
[ -n "${DRAFTER_EXPERTS_FP8_PREFIX:-}" ] &&
  ENV_ARGS+=(-e "VLLM_DRAFTER_EXPERTS_FP8_PREFIX=$DRAFTER_EXPERTS_FP8_PREFIX")

# Set when the (expanded) draft vocabulary is, or will be, mounted.
DV_ARGS=()
if [ "$print" = 1 ] || [ -r "$MTP_DRAFT_VOCAB" ]; then
  DV_ARGS=(-v "$MTP_DRAFT_VOCAB:/draft-vocab/ids.txt:ro"
           -e VLLM_MTP_DRAFT_VOCAB=/draft-vocab/ids.txt)
fi

# The whole command, assembled in a function so --print and --run cannot differ.
build_docker_run() {
  DOCKER_RUN=(docker run -d --name "$NAME" --restart unless-stopped \
    --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
    -v "$MODEL_DIR:/model:ro" -v "$TABLE_DIR:/ple-table:ro" \
    "${ENV_ARGS[@]}" \
    "${DV_ARGS[@]}" \
    "$IMAGE" \
    /model --served-model-name "${SERVED_NAME:-qwen3.8-flash-next}" \
      --host 0.0.0.0 --port 8000 --load-format "${LOAD_FORMAT:-fastsafetensors}" \
      --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
      $PC_ARG --enable-chunked-prefill --max-num-batched-tokens 8192 \
      $CC \
      $AT_ARG \
      --kv-cache-dtype auto \
      $EXTRA \
      --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3 \
      "${PIN_ARG[@]}" "${SPEC[@]}")
}

if [ "$print" = 1 ]; then
  build_docker_run
  printf '# effective container env\n'
  for ((i = 0; i < ${#ENV_ARGS[@]}; i += 2)); do
    printf '%s\n' "${ENV_ARGS[i + 1]}"
  done
  [ "${#DV_ARGS[@]}" -gt 0 ] && printf 'VLLM_MTP_DRAFT_VOCAB=%s\n' /draft-vocab/ids.txt
  printf '# effective host settings\n'
  printf 'NAME=%s\nIMAGE=%s\nMODEL_DIR=%s\nTABLE_DIR=%s\nPORT=%s\nMTP=%s\n' \
    "$NAME" "$IMAGE" "$MODEL_DIR" "$TABLE_DIR" "$PORT" "$MTP"
  printf 'MTP_DRAFT_VOCAB=%s\nDRAFT_VOCAB_SOURCE=%s\n' "$MTP_DRAFT_VOCAB" "$PWD/$vocab_gz"
  printf 'SPEC=%s\n' "${SPEC[*]:-}"
  printf '\n# docker run command\n'
  printf '%q ' "${DOCKER_RUN[@]}"; printf '\n'
  exit 0
fi

# --- preflight, then expand the draft vocabulary (upstream launch.sh --run) --
case "$IMAGE" in
  *:latest|latest|*:|'') echo 'a fixed image tag is required' >&2; exit 66 ;;
  *:*) ;;
  *) echo "IMAGE needs a tag: $IMAGE" >&2; exit 66 ;;
esac
docker image inspect "$IMAGE" >/dev/null || {
  echo "the ultrafast image is unavailable; run scripts/build-ultrafast.sh --run" >&2
  exit 66
}
[ -d "$MODEL_DIR" ] || { echo "missing model directory: $MODEL_DIR" >&2; exit 66; }
[ -d "$TABLE_DIR" ] || { echo "missing PLE table directory: $TABLE_DIR" >&2; exit 66; }

mkdir -p "$(dirname "$MTP_DRAFT_VOCAB")"
temp="$(mktemp "${MTP_DRAFT_VOCAB}.XXXXXX")"
trap 'rm -f "$temp"' EXIT
gzip -dc "$vocab_gz" > "$temp"
if [ "$(wc -l <"$temp")" -ne 65536 ] || [ "$(grep -c . <"$temp")" -ne 65536 ]; then
  echo 'draft vocabulary has the wrong length (want 65536 non-empty lines)' >&2
  exit 65
fi
mv "$temp" "$MTP_DRAFT_VOCAB"
trap - EXIT
chmod 644 "$MTP_DRAFT_VOCAB"
DV_ARGS=(-v "$MTP_DRAFT_VOCAB:/draft-vocab/ids.txt:ro"
         -e VLLM_MTP_DRAFT_VOCAB=/draft-vocab/ids.txt)
build_docker_run

# Refuse to clobber an existing container unless explicitly forced: a name
# collision is otherwise a silent stop+replace of whatever is running.
if [ "${ULTRAFAST_FORCE:-0}" != 1 ]; then
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
    echo "serve-ultrafast.sh: a container named $NAME already exists; use NAME=<other> ./scripts/serve-ultrafast.sh to run alongside, or ULTRAFAST_FORCE=1 to stop+replace it" >&2
    exit 1
  fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
"${DOCKER_RUN[@]}"

echo ">> $NAME starting on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM)"
echo ">> draft vocab K=65536, PLE fast path $PLE_FAST_PATH, drafter fp8 $DRAFTER_EXPERTS_FP8"
echo ">> follow with: docker logs -f $NAME"
