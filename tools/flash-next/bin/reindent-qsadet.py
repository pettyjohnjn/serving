#!/usr/bin/env python3
"""Adapt jschmied's qsadet_patch.py to a target file where the top-k anchor has moved.

Upstream vLLM moved the QSA top-k selection from a method in ops/qsa.py to a module-level
function in ops/qsa_indexer.py. The patch script hard-codes (a) the anchor's indent and (b) the
name of the enclosing function it hangs a `_qsadet_loaded` flag on. This rewrites both in a
private copy of the script. Usage: reindent-qsadet.py <qsadet_patch.py> <target.py> <out.py>
"""
import re, sys
script, target, out = sys.argv[1:4]
src = open(script).read(); tgt = open(target).read()
m = re.search(r"^( *)topk_op = \($", tgt, re.M)
if not m:
    sys.exit(f"reindent: no 'topk_op = (' in {target}")
want = m.group(1)
# enclosing function of the anchor in the target
encl = None
for dm in re.finditer(r"^( *)def (\w+)\(", tgt[:m.start()], re.M):
    if len(dm.group(1)) < len(want):
        encl = dm.group(2)
if not encl:
    sys.exit("reindent: could not find the function enclosing the anchor")
def reindent(block):
    lines = block.split("\n")
    have = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join((want + l[have:]) if l.strip() else l for l in lines)
def fix(name, text):
    mm = re.search(r"^%s = ('''|\"\"\")(.*?)\1" % name, text, re.S | re.M)
    if not mm:
        sys.exit(f"reindent: {name} block not found in {script}")
    return text[:mm.start(2)] + reindent(mm.group(2)) + text[mm.end(2):]
src = fix("ANCHOR", src)
src = fix("NEW", src)
old_holder = "qsa_select_paged_tokens"
if old_holder not in tgt and old_holder in src:
    src = src.replace(old_holder, encl)
    print(f"   det patch: flag holder {old_holder} -> {encl}")
open(out, "w").write(src)
print(f"   det patch re-indented to {len(want)} spaces for {target.rsplit('/', 1)[-1]} (enclosing: {encl})")
