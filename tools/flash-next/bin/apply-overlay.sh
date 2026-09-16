#!/usr/bin/env bash
# Apply overlay/*.diff and overlay/ple_mmap.py to the extracted image tree (qwen4_exp only).
# Idempotent: every target is restored from its pristine .orig copy first, then patched, so a
# re-run (or `fn build --force`) always ends in the same state. Clears the markers of the
# patches that build on top (6, 8) so apply-patches re-applies them.
#   env: SP FN_HOME MODPKG
set -uo pipefail
: "${SP:?}" "${FN_HOME:?}"
OV=$FN_HOME/overlay; MARK=$SP/.fn-patches; mkdir -p "$MARK"
[ "${MODPKG:-}" = qwen4_exp ] || { echo "  overlay: not for package '${MODPKG:-}', skipped"; exit 0; }
M=vllm/models/qwen4_exp/nvidia
[ -d "$SP/$M" ] || { echo "!! $SP/$M missing (wrong image?)"; exit 1; }
F=0
put_new () {  # src  dst-relative
  cp --no-preserve=all "$OV/$1" "$SP/$2" && cmp -s "$OV/$1" "$SP/$2" && echo "  new   $2" || { echo "!! copy of $2 failed"; F=1; }
}
apply () {  # diff-name  target-relative
  local d=$OV/$1 t=$SP/$2
  [ -f "$t" ] || { echo "!! target missing: $2"; F=1; return; }
  [ -f "$t.orig" ] || cp --no-preserve=all "$t" "$t.orig" || { echo "!! backup of $2 failed"; F=1; return; }
  cp --no-preserve=all "$t.orig" "$t"
  if (cd "$SP" && patch -p1 -s -N --no-backup-if-mismatch < "$d"); then echo "  ok    $2 ($1)"; else echo "!! $1 did not apply to $2"; F=1; fi
}
put_new ple_mmap.py                 "$M/ops/ple_mmap.py"
apply ple_layer.diff                "$M/ple_layer.py"
apply model_state.diff              "$M/model_state.py"
apply mtp.diff                      "$M/mtp.py"
apply ops_ple.diff                  "$M/ops/ple.py"
apply ops_qsa.diff                  "$M/ops/qsa.py"
apply qsa.diff                      "$M/qsa.py"
apply platforms_interface.diff      "vllm/platforms/interface.py"
apply modelopt.diff                 "vllm/model_executor/layers/quantization/modelopt.py"
PYTHONPATH=$SP PYTHONNOUSERSITE=1 python3.12 - "$SP" "$M" <<'PY' || F=1
import ast, sys; SP, M = sys.argv[1:3]
for f in [f"{M}/ple_layer.py", f"{M}/ops/ple_mmap.py", f"{M}/model_state.py", f"{M}/mtp.py", f"{M}/ops/ple.py",
          f"{M}/ops/qsa.py", f"{M}/qsa.py", "vllm/platforms/interface.py",
          "vllm/model_executor/layers/quantization/modelopt.py"]:
    ast.parse(open(f"{SP}/{f}").read())
print("  all overlaid modules parse")
PY
rm -f "$MARK/5" "$MARK/6" "$MARK/8"
[ "$F" = 0 ] && { touch "$MARK/overlay"; echo "  overlay applied (markers 6/8 cleared for re-apply)"; } || { rm -f "$MARK/overlay"; echo "!! overlay INCOMPLETE"; }
exit $F
