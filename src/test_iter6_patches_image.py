#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU image regression checks."""

import ast
import hashlib
import os
import subprocess
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
PATCH_DIR = os.environ.get("ITER6_PATCH_DIR", "/selftest/src")

LLG = SP + "/vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py"
SKINNY = SP + "/vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py"
TOPK_OPS = SP + "/vllm/v1/sample/ops/topk_topp_sampler.py"
STATES = SP + "/vllm/v1/worker/gpu/sample/states.py"
GPU_SAMPLER = SP + "/vllm/v1/worker/gpu/sample/sampler.py"
REJ = SP + "/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"

FAILURES = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        FAILURES.append(msg)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parses(path):
    ast.parse(open(path).read())
    return True


def rerun(script, targets):
    """Re-run a patch script; assert it is a no-op on every target file."""
    before = {p: sha(p) for p in targets}
    proc = subprocess.run(
        [sys.executable, os.path.join(PATCH_DIR, script)],
        capture_output=True,
        text=True,
    )
    check(proc.returncode == 0, "%s re-run exits 0 (stderr: %s)" % (
        script, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""))
    check(
        "already applied" in proc.stderr,
        "%s re-run reports 'already applied'" % script,
    )
    for p in targets:
        check(sha(p) == before[p], "%s unchanged by %s re-run" % (
            p.rsplit("/", 1)[-1], script))


# --------------------------------------------------------------------------- #
# Patch 1 -- R3, low-latency GEMM on SM12x + the PDL knob
# --------------------------------------------------------------------------- #
def test_llg():
    print("[1] patch_low_latency_gemm_sm110.py")
    src = open(LLG).read()
    check(parses(LLG), "low_latency_gemm.py parses")
    check("qwen38next_low_latency_sm12x" in src, "SM12x block present")
    check("qwen38next_low_latency_sm110_thor" in src,
          "Thor-specific sm110 marker present")
    check("QWEN38NEXT_LOW_LATENCY_GEMM" in src, "env gate present")
    check("_llg_validate" in src, "plan validator present")
    emitted = src.split("def _llg_add_tp1_plans")[1].split("extra = [")[1].split("]")[0]
    check(
        "(10240, 320)" not in emitted,
        "(10240, 320) is NOT emitted: K mod 128 == 64, and the kernel raises "
        "ValueError on that shape (the validator moves the refusal to build time)",
    )
    for shape in ("(320, 10240)", "(96, 2560)"):
        check(shape in emitted, "%s plan entry emitted" % shape)
    gate = src.split("def _llg_sm12x_enabled")[1].split("def _llg_validate")[0]
    check(
        '"QWEN38NEXT_LOW_LATENCY_GEMM", "0"' in gate.replace("'", '"'),
        "gate reads QWEN38NEXT_LOW_LATENCY_GEMM defaulting to 0",
    )
    check(
        'is_device_capability_family(120)' in gate
        and 'is_device_capability_family(110)' in gate,
        "gate widens to the SM 12.x (GB10) AND SM 11.x (Thor) families",
    )
    check(
        "raise RuntimeError" in src.split("def _llg_add_tp1_plans")[1],
        "an invalid plan entry raises rather than being emitted",
    )
    # iter6b F7: exactly one assertable install line, either way.
    check(src.count('_llg_logger.info("iter6 R3 disabled")') == 1,
          "R3 emits exactly one 'iter6 R3 disabled' line")
    check(src.count('"iter6 R3 installed: %d dispatch attachments, '
                    'accepted shapes %s, PDL=%s"') == 1,
          "R3 emits exactly one 'iter6 R3 installed: <count> ...' line")
    check(src.count("_llg_signal(0, ())") == 2 and
          src.count("_llg_signal(len(_llg_attached), set(_llg_attached))") == 1,
          "every exit of enable_qwen38next_low_latency_gemm signals exactly once")
    check("_llg_attached.append((weight.shape[0], weight.shape[1]))" in src,
          "the dispatch attachments are counted, with their (N, K)")
    check("_llg_init_logger(__name__)" in src,
          "R3 logs on the engine-core logger (vllm init_logger)")

    ssrc = open(SKINNY).read()
    check(parses(SKINNY), "skinny_gemm.py parses")
    check("qwen38next_llg_pdl" in ssrc, "PDL knob present")
    check("qwen38next_llg_pdl_sm110_thor" in ssrc,
          "Thor-specific PDL marker present")
    check("_QWEN38NEXT_LLG_PDL: bool | None = None" in ssrc, "PDL latch declared")
    check(
        '"QWEN38NEXT_LLG_PDL", "1"' in ssrc.replace("'", '"'),
        "PDL knob defaults to 1 (stock behaviour) and only 0 disables",
    )
    rerun("patch_low_latency_gemm_sm110.py", [LLG, SKINNY])


# --------------------------------------------------------------------------- #
# Patch 4 -- L5a, verify-path truncation
# --------------------------------------------------------------------------- #
def test_topk():
    print("[3] patch_verify_topk_pivot.py")
    for p in (TOPK_OPS, STATES, GPU_SAMPLER, REJ):
        check(parses(p), "%s parses" % p.rsplit("/", 1)[-1])
    ops = open(TOPK_OPS).read()
    check("_verify_topk_triton" in ops, "env latch present in topk_topp_sampler.py")
    check(
        '"VLLM_VERIFY_TOPK_TRITON", "0"' in ops.replace("'", '"'),
        "gate defaults to 0",
    )
    check(
        "if HAS_TRITON and logits.shape[0] >= 8:" in ops,
        "the ordinary decode path's >= 8 heuristic is UNCHANGED",
    )
    check(
        "prefer_triton and _verify_topk_triton" in ops,
        "prefer_triton is honoured only when the env is set",
    )
    check("logits.dtype == torch.float32" in ops, "fp32 precondition asserted")
    check("prefer_triton: bool = False" in open(STATES).read(),
          "states.py threads prefer_triton")
    check("prefer_triton: bool = False" in open(GPU_SAMPLER).read(),
          "gpu/sample/sampler.py threads prefer_triton")
    check("prefer_triton=True" in open(REJ).read(),
          "rejection_sampler.py:_verify passes prefer_triton=True")
    # iter6b F7.
    check(ops.count('logger.info("iter6 L5a disabled")') == 1,
          "L5a emits exactly one 'iter6 L5a disabled' line")
    check(ops.count('"iter6 L5a installed: 1 sampler hook on apply_top_k_top_p "')
          == 1,
          "L5a emits exactly one 'iter6 L5a installed: 1 sampler hook ...' line")
    sig = ops.split("iter6 L5a installed")[1].split("else:")[0]
    check("fp32" in sig,
          "the L5a install line names the fp32 precondition")
    rerun("patch_verify_topk_pivot.py", [TOPK_OPS, STATES, GPU_SAMPLER, REJ])


def main():
    for name in (
        "QWEN38NEXT_LOW_LATENCY_GEMM",
        "QWEN38NEXT_LLG_PDL",
        "VLLM_VERIFY_TOPK_TRITON",
    ):
        os.environ.pop(name, None)
    test_llg()
    test_topk()
    if FAILURES:
        print("\n%d FAILURES:" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("\nall iteration-6 in-image patch checks passed")


main()
