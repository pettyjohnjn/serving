#!/usr/bin/env bash
# Content checks. Deliberately does NOT trust the marker files: it looks for the
# actual edits, so a re-extracted image or an upstream change is caught.
#   env: SP MODPKG KDETDIR FN_HF FN_MODEL MODE   (SNAP is resolved here, on the node)
set -uo pipefail
F=0
SNAP=$(ls -d "$FN_HF/hub/models--${FN_MODEL//\//--}"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1)
SNAP=${SNAP%/}
chk () { if eval "$2" >/dev/null 2>&1; then echo "  ok   $1"; else echo "  FAIL $1"; F=$((F+1)); fi; }
# Patch checks honour the skip list (FN_SKIP_PATCHES): "chkp N label test" is skipped when N is listed.
chkp () { case " ${SKIP:-} " in *" $1 "*) echo "  skip $2  (patch $1 not applied on this tree)";; *) chk "$2" "$3";; esac; }
M=$SP/vllm/models/$MODPKG/nvidia
chk "image extracted ($MODPKG)"        "[ -f '$M/ple_layer.py' ]"
chkp 1 "1 ple-mmap hook in ple_layer.py"  "grep -q '_ple_mmap_apply' '$M/ple_layer.py'"
chkp 1 "1 ple-mmap module present"        "[ -f '$SP/vllm_ple_mmap.py' ]"
if [ "$MODPKG" = qwen4_exp ]; then
  chk "overlay ple_mmap module present"    "[ -f '$M/ops/ple_mmap.py' ]"
  chk "overlay hooks in ple_layer.py"       "grep -q '_ple_mmap_enabled' '$M/ple_layer.py'"
  chk "overlay modelopt MTP fixes"          "grep -q '_quantized_layer_group_size' '$SP/vllm/model_executor/layers/quantization/modelopt.py'"
  chk "overlay PR #54846 in qsa.py"         "grep -q '_nvfp4_attention_spec' '$M/qsa.py'"
  chk "overlay PR #54846 in ops/qsa.py"     "grep -q '_nvfp4_decode_e2m1' '$M/ops/qsa.py'"
fi
chk "2 fla shared-mem gate 101376"     "grep -q 101376 '$SP/vllm/third_party/flash_linear_attention/ops/utils.py'"
chk "2 fla num_warps pinned to [2]"    "grep -qE 'for num_warps in \\[2\\]' '$SP/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py'"
chk "3 mamba_utils guarded"            "grep -q 'out-of-range' '$SP/vllm/v1/worker/mamba_utils.py'"
chkp 4 "4 prefix-cache block_size fix"    "grep -q 'mamba_block_size or' '$SP/vllm/v1/worker/gpu/model_states/mamba_hybrid.py'"
chkp 4 "4 scheduler block_size fix"       "grep -q 'scheduler block size' '$SP/vllm/v1/core/sched/scheduler.py'"
chkp 5 "5 qsa exact top-k"                "grep -qi 'exact' '$M/ops/qsa.py'"
chk "6 fp8-hybrid hook in modelopt"    "grep -q '_fp8_hybrid_apply' '$SP/vllm/model_executor/layers/quantization/modelopt.py'"
chk "6 fp8-hybrid dispatch in qsa"     "grep -q '_fp8_hybrid_excluded(quant_config)' '$M/qsa.py'"
chkp 7 "7 qsa fp8 KV path"                "grep -qi 'fp8' '$M/ops/qsa.py'"
chk "8 det-topk kernel built"          "[ -s '$KDETDIR/_C_det.so' ]"
QSA_DET=$M/ops/qsa.py; [ -f "$M/ops/qsa_indexer.py" ] && grep -q "torch.ops._C.persistent_topk" "$M/ops/qsa_indexer.py" && QSA_DET=$M/ops/qsa_indexer.py
chk "8 det-topk wired into qsa"        "grep -q 'QSADET' '$QSA_DET'"
chk "every module still parses"        "PYTHONPATH='$SP' PYTHONNOUSERSITE=1 python3.12 -c \"
import ast,sys
for f in ['$M/ple_layer.py','$M/qsa.py','$M/ops/qsa.py','$SP/vllm/v1/worker/mamba_utils.py',
          '$SP/vllm/v1/core/sched/scheduler.py','$SP/vllm/model_executor/layers/quantization/modelopt.py']:
    ast.parse(open(f).read())\""
chk "checkpoint present"               "[ -n '$SNAP' ] && [ -d '$SNAP' ]"
if [ "$MODE" = hybrid ]; then
  chk "hybrid checkpoint prepared"     "[ -f '${SNAP}-fp8hybrid/.prepared' ]"
  chk "hybrid has fp8 side tensors"    "PYTHONPATH='$SP' python3.12 -c \"
import json;m=json.load(open('${SNAP}-fp8hybrid/model.safetensors.index.json'))['weight_map']
n=sum(1 for k in m if k.endswith('weight_scale_inv'))
assert n>0, 'no fp8 tensors'; print(n)\""
fi
exit $F
