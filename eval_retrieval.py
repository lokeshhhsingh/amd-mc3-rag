#!/usr/bin/env python3
"""Check the reading + search half of your solution against the sample questions.

No GPU, no Docker, no model needed:

    python3 eval_retrieval.py mc3-corpus
    python3 eval_retrieval.py mc3-corpus-hard hard-questions.json

It builds the index, prints what could and could not be read, then for every sample
question shows what retrieval hands to the model and whether the files a correct answer
needs are all in that set. (Answer wording and final citations are the model's job.)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "starter" / "app"))
os.environ.setdefault("MC3_INDEX_DIR", tempfile.mkdtemp(prefix="mc3idx-"))

import app                    # noqa: E402
from rag import llm, search   # noqa: E402


def main() -> int:
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "mc3-corpus").resolve()
    app.INDEX_DIR = Path(os.environ["MC3_INDEX_DIR"])
    print(f"running as uid {os.geteuid()}  (a non-root uid is what makes the unreadable file truly fail)\n")
    app.index(corpus)
    idx = search.load_index(app.INDEX_DIR)

    print("\nFILES")
    for m in idx["manifest"]:
        print(f"  {m['status']:<10} {m['chunks']:>3} chunks  {m['file']}")

    qfile = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "sample-questions.json"
    qs = json.loads(qfile.read_text())["queries"]
    covered = 0
    print("\nQUESTIONS")
    for q in qs:
        cands = search.retrieve(idx, q["query"])
        files = []
        for c in cands:
            if c["file"] not in files:
                files.append(c["file"])
        need = q["expected_citations"]
        shown_files = {c["file"] for c in llm.select_candidates(cands)}
        ok = all(f in files for f in need) and all(f in shown_files for f in need)   # retrieved AND within the prompt budget
        covered += ok
        ranks = {f: files.index(f) + 1 for f in need if f in files}
        flag = "OK  " if ok else "MISS"
        print(f"\n{flag} Q{q['n']}: {q['query']}")
        print(f"     needs: {need or '(refusal: nothing)'}  found at file-rank: {ranks or '-'}")
        print(f"     retrieved files: {files[:6]}")
        hop = [c for c in cands if c["via"].startswith("hop")]
        if hop:
            print(f"     via second hop: {[(c['file'], c['via']) for c in hop]}")
    print(f"\nretrieval covers {covered}/{len(qs)} questions (needed files retrieved AND shown to the model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
