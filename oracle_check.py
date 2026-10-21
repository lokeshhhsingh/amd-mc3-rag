#!/usr/bin/env python3
"""Tests the CITATION logic alone, with a pretend model that always gets the answer right.

For every question it plays two model behaviours:
  all  - cites every shown excerpt that contains the answer
  best - cites only the best-ranked one (or, for image answers, the image excerpt)
and compares the files that come out with the expected citation set. Mismatches here are bugs in my logic,
not in the model.     python3 oracle_check.py mc3-corpus-hard hard-questions.json
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

import app                   # noqa: E402
from rag import llm, search  # noqa: E402


class Oracle:
    def __init__(self, answer, evidence): self.reply = json.dumps({"answer": answer, "evidence": evidence})
    def complete(self, prompt, images, max_new_tokens=160): return self.reply


def main() -> int:
    corpus = Path(sys.argv[1]).resolve()
    qfile = Path(sys.argv[2])
    app.INDEX_DIR = Path(os.environ["MC3_INDEX_DIR"])
    sys.stderr = open(os.devnull, "w")
    app.index(corpus)
    sys.stderr = sys.__stderr__
    idx = search.load_index(app.INDEX_DIR)
    bad = 0
    for q in json.loads(qfile.read_text())["queries"]:
        if not q["expected_answer"]:
            continue
        cands = search.retrieve(idx, q["query"])
        shown = llm.select_candidates(cands)
        holders = [i for i, c in enumerate(shown, 1) if c["kind"] == "text" and llm.contains_answer(q["expected_answer"], c["text"])]
        images = [i for i, c in enumerate(shown, 1) if c["kind"] == "image" and c["file"] in q["expected_citations"]]
        image_answer = all(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in q["expected_citations"])
        variants = {"all": images if image_answer else holders, "best": images if image_answer else holders[:1]}
        for name, ev in variants.items():
            out = llm.answer_with_llm(Oracle(q["expected_answer"], ev), q["query"], cands, corpus)
            if sorted(out[1]) != sorted(q["expected_citations"]):
                bad += 1
                print(f"Q{q['n']:>2} [{name}] got {out[1]}\n        want {q['expected_citations']}   ({q['query'][:70]})")
    print(f"\ncitation mismatches with a perfect model: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
