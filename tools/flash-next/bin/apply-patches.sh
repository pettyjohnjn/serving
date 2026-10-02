#!/usr/bin/env bash
# Apply the patch stack to the extracted image tree. Idempotent: each step is
# marker-guarded, and every edit asserts its anchor so a moved upstream fails
# loudly instead of silently no-opping. Runs on the compute node.
#   env: SP REPO MODPKG [KDET KDETDIR DET_ARCH BUILD_KERNEL SKIP]
set -uo pipefail
: "${SP:?}" "${MODPKG:?}"
# REPO holds the patch sources; the kernel build (BUILD_KERNEL=1) fetches its own and
# needs none, so only insist on it for patches 1-7.
[ "${BUILD_KERNEL:-0}" = 1 ] || : "${REPO:?}"
py () { PYTHONPATH="$SP" PYTHONNOUSERSITE=1 python3.12 "$@"; }
MARK=$SP/.fn-patches; mkdir -p "$MARK"
M=$SP/vllm/models/$MODPKG/nvidia
PLE=$M/ple_layer.py; QSA_OPS=$M/ops/qsa.py; QSA_TOP=$M/qsa.py
# vLLM main moved the QSA top-k (the det-kernel anchor) from ops/qsa.py into ops/qsa_indexer.py
# as a module-level function (Sep 2026, qwen4_exp). Patch 8 goes to whichever file holds it.
QSA_DET_FILE=$QSA_OPS
[ -f "$M/ops/qsa_indexer.py" ] && grep -q "torch.ops._C.persistent_topk" "$M/ops/qsa_indexer.py" && QSA_DET_FILE=$M/ops/qsa_indexer.py
MU=$SP/vllm/v1/worker/mamba_utils.py
MO=$SP/vllm/model_executor/layers/quantization/modelopt.py
FLA_U=$SP/vllm/third_party/flash_linear_attention/ops/utils.py
FLA_C=$SP/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
# SKIP="1 7": leave those patch numbers out, e.g. when an overlay already covers them.
step () {
  case " ${SKIP:-} " in *" $1 "*) echo "  -- $1 $2 (skipped via SKIP)"; return 1 ;; esac
  [ -f "$MARK/$1" ] && { echo "  -- $1 $2 (already)"; return 1; }
  echo "  >> $1 $2"; return 0
}
# class name follows the package name: qwen3_8_flash_next -> Qwen3_8FlashNext...
case "$MODPKG" in
  qwen3_8_flash_next) NGRAM_CLS=Qwen3_8FlashNextNGramEmbedding ;;
  qwen4_exp)          NGRAM_CLS=Qwen4ExpNGramEmbedding ;;
  *) echo "!! unknown model package $MODPKG"; exit 1 ;;
esac

if [ "${BUILD_KERNEL:-0}" != 1 ]; then
  if step 1 "ple-mmap"; then
    cp "$REPO/src/vllm_ple_mmap.py" "$SP/vllm_ple_mmap.py"
    cp --update=none "$PLE" "$PLE.orig" 2>/dev/null || true
    printf '\n\n# --- fn: PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(%s)\n' "$NGRAM_CLS" >> "$PLE"
    py -c "import ast;ast.parse(open('$PLE').read())" && touch "$MARK/1"
  fi
  if step 2 "fla-gb10"; then
    sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # fn: GB10 reports 99KiB shared mem|' "$FLA_U"
    grep -q 101376 "$FLA_U" || { echo "!! fla shmem anchor missing"; exit 1; }
    sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # fn: fla#953 Blackwell tl.dot race|' "$FLA_C"
    grep -q "fn: fla#953" "$FLA_C" || { echo "!! fla warps anchor missing"; exit 1; }
    touch "$MARK/2"
  fi
  if step 3 "mamba-utils-guarded"; then
    cp --update=none "$MU" "$MU.orig" 2>/dev/null || true
    cp "$REPO/src/mamba_utils_guarded.py" "$MU"
    py -c "import ast;ast.parse(open('$MU').read())" && touch "$MARK/3"
  fi
  if step 4 "prefix-cache block_size"; then
    py "$REPO/src/patch_mamba_block_size.py" "$SP" && touch "$MARK/4"
  fi
  if step 5 "qsa exact top-k"; then
    py "$REPO/src/patch_qsa_exact_topk.py" "$QSA_OPS" && touch "$MARK/5"
  fi
  if step 6 "fp8 hybrid dispatch"; then
    cp "$REPO/src/vllm_fp8_hybrid_modelopt.py" "$SP/vllm_fp8_hybrid_modelopt.py"
    cp --update=none "$MO" "$MO.orig" 2>/dev/null || true
    printf '\n\n# --- fn: NVFP4 experts + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> "$MO"
    cp --update=none "$QSA_TOP" "$QSA_TOP.orig" 2>/dev/null || true
    sed -i 's/quant_config=model\.without_modelopt_fp4(quant_config)/quant_config=_fp8_hybrid_excluded(quant_config)/' "$QSA_TOP"
    sed -i 's/^from \. import model$/from . import model\nfrom vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded/' "$QSA_TOP"
    grep -q "_fp8_hybrid_excluded(quant_config)" "$QSA_TOP" || { echo "!! qsa dispatch anchor missing"; exit 1; }
    py -c "import ast;ast.parse(open('$MO').read());ast.parse(open('$QSA_TOP').read())" && touch "$MARK/6"
  fi
  if step 7 "qsa fp8 KV"; then
    py "$REPO/src/patch_qsa_fp8_kv.py" "$SP" && touch "$MARK/7"
  fi
  exit 0
fi

# ---- patch 8: deterministic top-k kernel (needs nvcc + a GPU) ----------------
: "${KDET:?}" "${KDETDIR:?}" "${DET_ARCH:?}"
if step 8 "deterministic top-k kernel"; then
  mkdir -p "$KDETDIR/src"
  for f in build_det.py bindings_det.cpp topk_det.cu torch_utils.h persistent_topk.cuh; do
    curl -fsSL -o "$KDETDIR/src/$f" "$KDET/patches/kernel-det/$f" || exit 1
  done
  curl -fsSL -o "$KDETDIR/qsadet_patch.py" "$KDET/tools/determinism/qsadet_patch.py" || exit 1
  # torch's cpp_extension shells out to a bare `ninja`; find one.
  # ninja >= 1.13 installs the binary into the image's bin/ instead of the
  # wheel's ninja/data/bin/, so check both layouts before the fallbacks.
  NINJA=""
  for c in "${SP%/lib/python*/dist-packages}/bin" "$SP/ninja/data/bin" \
           "${FN_SCRATCH:-}/venv/bin" /scratch/fn2/venv/bin \
           "$HOME/serving/venv-webui/bin"; do
    [ -n "$c" ] && [ -x "$c/ninja" ] && NINJA=$c && break
  done
  # last resort: whatever is already on PATH
  [ -z "$NINJA" ] && command -v ninja >/dev/null 2>&1 && NINJA=$(dirname "$(command -v ninja)")
  [ -n "$NINJA" ] || { echo "!! no ninja binary for the kernel build"; exit 1; }
  echo "   ninja: $NINJA/ninja ($("$NINJA/ninja" --version 2>/dev/null))"
  ( cd "$KDETDIR/src" && DET_BUILD_DIR="$KDETDIR/build" DET_ARCH="$DET_ARCH" \
      PATH="$NINJA:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
      PYTHONPATH="$SP" PYTHONNOUSERSITE=1 python3.12 build_det.py 2>&1 | tail -3 ) || exit 1
  cp "$KDETDIR/build/_C_det.so" "$KDETDIR/_C_det.so"
  # The upstream patch script hard-codes the anchor's indent and enclosing-function name from the
  # old ops/qsa.py; bin/reindent-qsadet.py adapts a private copy to the actual target.
  py "$FN_HOME/bin/reindent-qsadet.py" "$KDETDIR/qsadet_patch.py" "$QSA_DET_FILE" "$KDETDIR/qsadet_patch.local.py" || exit 1
  VLLM_QSA_PY="$QSA_DET_FILE" py "$KDETDIR/qsadet_patch.local.py" || exit 1
  py -c "import ast;ast.parse(open('$QSA_DET_FILE').read())" && touch "$MARK/8"
fi
