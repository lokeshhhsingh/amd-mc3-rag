"""One-time patch: when the answer comes from a chunk found by following an identifier (a second hop),
also cite the chunk where that identifier was found. Safe to run once; it refuses to run twice.
Usage:  python3 patch_multihop.py [path/to/starter/app/rag]
"""
import sys
from pathlib import Path

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "starter" / "app" / "rag"

s = (base / "search.py").read_text()
if '"src_doc"' in s:
    sys.exit("already patched")
a = 'chosen[hdid] = {"score": hs, "via": f"hop:{ident}"}'
b = '"score": round(meta["score"], 3), "via": meta["via"]})'
assert a in s and b in s, "search.py is not the expected version"
s = s.replace(a, 'chosen[hdid] = {"score": hs, "via": f"hop:{ident}", "src": did}')
s = s.replace(b, '"score": round(meta["score"], 3), "via": meta["via"], "src_doc": meta.get("src")})')
(base / "search.py").write_text(s)

l = (base / "llm.py").read_text()
c = '    files = sorted({cands[i - 1]["file"] for i in ev})'
d = "answer impossible. If the answer needs two (one gives an id, another gives the value for that id), list both."
assert c in l and d in l, "llm.py is not the expected version"
l = l.replace(c, '''    by_doc = {c.get("doc"): i for i, c in enumerate(cands, start=1)}
    for i in list(ev):                        # evidence found by following an id also needs the excerpt that named the id
        src = cands[i - 1].get("src_doc")
        if src is not None and src in by_doc:
            ev.append(by_doc[src])
    files = sorted({cands[i - 1]["file"] for i in set(ev)})''')
l = l.replace(d, d + "\n   Also list any excerpt that only told you which id, code or part number to look up in another excerpt.")
(base / "llm.py").write_text(l)
print("patched: llm.py and search.py")
