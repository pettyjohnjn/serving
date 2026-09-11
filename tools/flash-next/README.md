# Flash-Next runtime: build and verify

The `qwen38-flash-next` model profile needs a patched vLLM tree, a compiled kernel
and a converted checkpoint on the serving node's local disk. This directory builds
them. It is the Flash-Next counterpart of `bin/fetch_model.sh`: a node-prep step,
run by an operator, before the profile is selected.

The point of the layout is that it is **repeatable**: every stage is idempotent,
every cluster-specific value is in one file, and verification checks the actual
file contents rather than trusting stamp files.

## Use

    tools/flash-next/bin/fn build      # image + patches + kernel + weights + hybrid checkpoint
    tools/flash-next/bin/fn verify     # 18 content checks; must be green before serving

then, from the production stack:

    serving restart qwen38-flash-next
    serving test                       # 8/8 is the acceptance gate
    serving restart qwen38-27b         # roll back; both sets of weights coexist

`fn build` submits the 126 GiB checkpoint download as a job and returns; run it
again once that finishes to do the conversion. Everything it produces is owned by
whoever ran it and is world-readable, which is what lets a service account serve
artifacts an operator built; if that account should own them outright, `chown -R`
the two scratch trees in `etc/fn.env` after `verify` is green. It exits non-zero and names the
stage if anything failed. The profile refuses to launch until `verify` would pass,
so a half-built tree cannot be served by accident.

## When the cluster changes

Edit `etc/fn.env` only: node, partition, scratch paths, and every pinned input
(image digest, patch commit, kernel SHA, GPU arch, checkpoint). Then
`fn build && fn verify`. Re-running build is always safe.

If `/scratch` is wiped, `fn build` rebuilds everything and re-submits the download.
If a patch stops applying, `verify` names the check that failed and `fn build --force`
re-applies the stack from scratch.

## Layout

    etc/fn.env            all tunables + pinned versions
    bin/fn                driver (build / verify / clean)
    bin/apply-patches.sh  the 8 patches, marker-guarded, anchors asserted
    bin/verify.sh         content checks, run on the node
    flash-dgx/            clone of the patch sources (created by build; git-ignored)
    logs/                 download job output (git-ignored)

## Traps this layout exists to survive

- **`fn build` used to print "build done" after a failed stage.** Stage results are
  now collected and the summary is honest; do not trust an older copy's summary.
- **ninja moved.** ninja >= 1.13 installs its binary into the image's `bin/`, not the
  wheel's `ninja/data/bin/`. The kernel stage checks both, then `PATH`.
- **Anything started from an `srun --overlap` step dies when that step exits.**
  Long stages are submitted as jobs, not run inside an overlap step.
- **Copying from a read-only mirror with `cp -a` preserves the read-only bits
  mid-copy.** Use `cp -r` then `chmod -R u+w`.
- **`pkill -f` matches the calling script's own command line.** Use `[p]attern`.
- **vLLM main renamed `qwen3_8_flash_next` to `qwen4_exp`** and changed the
  `NGramEmbedding` constructor the PLE patch hooks. `FN_MODEL_PKG` selects which;
  the patch script picks the matching class name.
- **Page-cache starvation.** A job cgroup is charged for file pages it reads, so a
  126 GiB weight read can push a memory-capped job to the OOM killer before the model
  is even loaded. The profile submits with `--mem=0` (whole node) for this reason.
