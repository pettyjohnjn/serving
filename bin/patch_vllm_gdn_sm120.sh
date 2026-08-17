#!/bin/bash
# patch_vllm_gdn_sm120.sh — let vLLM use FlashInfer's SM120 GDN prefill kernel on GB10.
#
#   bash bin/patch_vllm_gdn_sm120.sh            apply
#   bash bin/patch_vllm_gdn_sm120.sh --revert   restore the stock file
#   bash bin/patch_vllm_gdn_sm120.sh --status   show which kernel path is enabled
#
# WHY
# Qwen3.8-27B is 48/64 linear-attention (Gated DeltaNet) layers, so GDN prefill dominates
# time-to-first-token. vLLM 0.27.1 picks the fast FlashInfer GDN kernel only for:
#     capability 90 (Hopper), or capability FAMILY 100 (datacenter Blackwell) + head_k_dim 128
# GB10 is capability (12, 1) -- the SM120 family -- so it fails both tests and silently falls
# back to the Triton/FLA kernel. Measured consequence: prefill at 1,047-2,097 tok/s, i.e.
# ~50-100 TFLOPS against a part that should do far better.
#
# But FlashInfer *does* ship an SM120 path. Verified on this machine:
#     flashinfer.gdn_prefill.chunk_gated_delta_rule_sm120   AVAILABLE
# and flashinfer/gdn_prefill.py dispatches on compute-capability MAJOR
# (`elif _arch_major == 12: chunk_gated_delta_rule_sm120(...)`), which is 12 for GB10.
# So the kernel is present and reachable; only vLLM's gate excludes it.
#
# This patch widens that gate to include the SM120 family. It changes kernel selection only --
# no numerics are altered by the patch itself. NOTE: the SM120 path requires the recurrent
# state in float32 (FlashInfer docs: "float32 on SM90/SM120; the SM100 path also accepts
# bfloat16"), so do NOT combine this with --mamba-ssm-cache-dtype bfloat16.
#
# Re-apply after any `uv pip install`/upgrade of vLLM -- site-packages edits do not survive.

set -uo pipefail

ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
TARGET="$ROOT/venv2/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
BACKUP="$TARGET.orig"
MARK="vllm-gdn-sm120-patch"

[ -f "$TARGET" ] || { echo "cannot find $TARGET"; exit 1; }

case "${1:-apply}" in
--status)
    if grep -q "$MARK" "$TARGET"; then echo "PATCHED: SM120 family enabled for FlashInfer GDN prefill"
    else echo "STOCK: SM120 family falls back to Triton/FLA"; fi
    grep -n "is_device_capability_family" "$TARGET" | sed 's/^/  /'
    exit 0 ;;
--revert)
    [ -f "$BACKUP" ] || { echo "no backup at $BACKUP"; exit 1; }
    cp "$BACKUP" "$TARGET"; echo "reverted to stock vLLM"; exit 0 ;;
apply) ;;
*) echo "usage: $0 [apply|--revert|--status]"; exit 2 ;;
esac

if grep -q "$MARK" "$TARGET"; then echo "already patched; nothing to do"; exit 0; fi
[ -f "$BACKUP" ] || cp "$TARGET" "$BACKUP"

python3 - "$TARGET" "$MARK" <<'PY'
import sys
target, mark = sys.argv[1], sys.argv[2]
s = open(target).read()

old = """    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
"""
new = f"""    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        # {mark}: GB10 / RTX Blackwell report capability (12, x). Stock vLLM excludes the
        # whole SM120 family here and falls back to Triton/FLA, but FlashInfer ships
        # chunk_gated_delta_rule_sm120 and dispatches on capability major == 12, so the
        # fast path is available. Requires float32 recurrent state on this arch.
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
"""
assert old in s, "gate block not found -- vLLM version changed; re-check the patch"
open(target, "w").write(s.replace(old, new, 1))
print("patched gate to include SM120 family")
PY

python3 -c "import ast,sys; ast.parse(open('$TARGET').read()); print('syntax OK')" || {
    echo "patched file failed to parse; reverting"; cp "$BACKUP" "$TARGET"; exit 1; }
echo "done -- restart the service for it to take effect:  serving restart"
