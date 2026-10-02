# overlay/ — the Flash-Next PLE overlay for the vLLM nightly tree

Applied by `bin/apply-overlay.sh` (build stage 3) on the extracted image, before the patch
stack. Only for `FN_MODEL_PKG=qwen4_exp` (vLLM nightly `8a728663`, 2026-09-04). Each `.diff`
is a unified diff against the pristine file of that image; `ple_mmap.py` is a new file.

What it does: serves the 47.7 GiB PLE n-gram table from the NVMe copy of the checkpoint
through an mmap gather (`QWEN4EXP_PLE_MMAP=1`), so the table does not live in unified
memory. With the nightly's kernels this is +22-29% decode over the August image. Also
included: two fixes so the NVIDIA MTP head loads on this nightly (`modelopt.diff`), the
staged-gather / reduced-vocab draft paths (`model_state.diff`, `mtp.diff`; not used by the
production profile but part of the tested file set), and two upstream PRs that had not
landed in the pinned nightly.

| file | adds | origin |
|---|---|---|
| `ple_mmap.py` (new: `nvidia/ops/ple_mmap.py`) | 241 lines | Kai / 2Wild (tonyd2wild), Apache-2.0 |
| `ple_layer.diff` | +195 | tonyd2wild: mmap / staged / resident hooks in `nvidia/ple_layer.py` |
| `model_state.diff`, `mtp.diff` | +34, +56 | tonyd2wild: staged PLE gather, reduced-vocab draft |
| `modelopt.diff` | +47 | tonyd2wild: draft-local MTP layer index + 128x128 block-scale branch |
| `ops_ple.diff` | +8 | vLLM PR #55375 (peakcrosser7): fused-PLE stride fix |
| `ops_qsa.diff`, `qsa.diff`, `platforms_interface.diff` | +267, +252, +21 | vLLM PR #54846 (andreasgru): QSA fp8 / nvfp4 KV cache |

Source: https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark (Apache-2.0), the
`single-spark-vllm-tp1/patch/` tree as of 2026-09-05, with the two PR overlays it carries.
The diffs were generated on 2026-09-16 against the image files and dry-run verified.

Licensing: vLLM and both PRs are Apache-2.0 (Copyright contributors to the vLLM project);
the tonyd2wild files are Apache-2.0. This directory redistributes modified copies under the
same license; see the SPDX headers inside the diffs.

Order matters: our patches 6 (fp8-hybrid dispatch) and 8 (deterministic top-k) modify
`modelopt.py` and `ops/qsa*.py` after the overlay, so `apply-overlay.sh` clears their
markers and `apply-patches.sh` re-applies them. Patches 1, 4, 5 and 7 are skipped on this
tree (`FN_SKIP_PATCHES`, defaulted in `bin/fn`): 1 is replaced by the overlay's mmap path,
4 by the nightly's own block-size fix, 5 has no anchor and is unneeded, 7 is inside PR #54846.
