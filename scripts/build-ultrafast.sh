#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the Thor ultrafast image on top of this repo's base image.
#
#   scripts/build-ultrafast.sh              # print the commands, build nothing
#   scripts/build-ultrafast.sh --run        # build for real
#
# Mirrors the upstream recipe's image build script: square away the base image
# first (only if its tag is not already present locally), then build
# Dockerfile.ultrafast on top of it. The v16b regression suites run INSIDE the
# ultrafast build as RUN steps, so a green build is a green test run; there is
# no separate test phase here.
#
# Override the result tag with IMAGE=..., the base with BASE_IMAGE=...
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$here/Dockerfile" ] && [ -f "$here/Dockerfile.ultrafast" ] || {
  echo 'missing a Dockerfile' >&2; exit 66;
}

# The base tag the existing serve scripts use (scripts/serve-intel-ar.sh).
base_image="${BASE_IMAGE:-qwen38-flash-dgx}"
# <base-name>:ultrafast-thor-20261007, base name taken from BASE_IMAGE.
image="${IMAGE:-${base_image%%:*}:ultrafast-thor-20261007}"

# Fail before pulling or building anything if a build input is not present.
missing=()
for name in \
  src/vllm_ple_mmap.py src/vllm_fp8_hybrid.py \
  src/draft_vocab_common.py src/vllm_mtp_draft_vocab.py \
  src/patch_verify_topk_pivot.py src/patch_low_latency_gemm_sm110.py \
  src/patch_block_drop.py src/blockdrop_base_sha256.json \
  src/test_iter6_patches_image.py src/test_block_drop_cpu.py \
  src/test_draft_vocab_cpu.py config/ultrafast/draft-vocab-ids-K65536.txt.gz
do
  [ -f "$here/$name" ] || missing+=("$name")
done
if [ "${#missing[@]}" -ne 0 ]; then
  printf 'Missing ultrafast build input(s):\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 66
fi

commands=()
if ! docker image inspect "$base_image" >/dev/null 2>&1; then
  commands+=("docker build -f '$here/Dockerfile' -t '$base_image' '$here'")
fi
commands+=("docker build --pull=false --build-arg BASE_IMAGE='$base_image' -f '$here/Dockerfile.ultrafast' -t '$image' '$here'")

if [ "$mode" = --print ]; then
  printf '%s\n' "${commands[@]}"
  exit 0
fi

for cmd in "${commands[@]}"; do
  eval "$cmd"
done
docker image inspect "$image" --format '{{.Id}}'
