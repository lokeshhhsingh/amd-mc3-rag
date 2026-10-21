#!/usr/bin/env python3
"""Reference submission skeleton for AMD Mini-Challenge 3 (RAG).

Unlike MC-2, the harness invokes this script in TWO different ways, inside your
already-running container.

1. ONCE, before any graded clock starts, to build whatever index you want:

       python3 /app/app.py --index /app/corpus

   This pass is ungraded and charged to your startup budget, not to a query.
   It is where you should pay for parsing PDFs, spreadsheets and images.

2. ONCE PER QUESTION, ten times:

       python3 /app/app.py --corpus /app/corpus \\
                           --query-id query_01 \\
                           --query "What is the maximum junction temperature?"

   and then it reads the file you wrote:

       /app/output/query_01_output.json

   The query id is GIVEN to you, not derived -- a question has no filename to
   take a stem from. Write to exactly that id plus _output.json.

The output is a JSON object with three keys:

    {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}

  * answer     -- the VALUE only. "94", not "The maximum junction temperature
                  is 94 C". Qualifiers that are part of the value stay:
                  "Q3 FY27", not "Q3". Units and degree signs are normalised
                  away for you, so 94, "94 C" and "94°C" are all accepted.
  * citations  -- corpus-relative paths, compared as an EXACT SET. The test is
                  necessity, not relevance: cite a file only if REMOVING it
                  would make your answer impossible. Dumping your whole
                  retrieval fails almost every question even when the answer is
                  right. Some questions need two files; cite both.
  * confidence -- recorded, never scored.

Missing "answer" or missing "citations" is malformed and scores zero, on
purpose: a crashed writer and a considered refusal must not look the same.

If the corpus does not contain the answer, return an empty answer AND an empty
citation list. "I don't know" is prose, and prose is graded as a wrong answer.
Guessing is wrong, and so is answering from what the model already knows --
the products in this corpus are fictional, so anything recalled rather than
retrieved is wrong by construction.

Three things that will cost you points no matter how good your model is:

  * Loading the model inside answer(). The harness starts a NEW PROCESS for
    every question. A naive implementation loads ten times and blows the
    budget. Persist your index in --index, and keep the model resident behind
    a unix socket, or make loading cheap enough not to matter.
  * Letting one bad file kill the walk. The corpus deliberately contains an
    unreadable file, an encrypted PDF, an unknown binary type and an empty
    directory. A walk that raises on the first unreadable file indexes nothing
    after it -- and because walk order follows the directory listing, WHICH
    files you lose depends on filename order. That is how a submission tests
    clean locally and grades badly here.
  * Running on the CPU. VRAM is sampled continuously and a run that never uses
    the GPU is rejected, so place your model on the device explicitly rather
    than letting it fall back.

Replace index() and answer() below. Leave the plumbing alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


def index(corpus: Path) -> None:
    """Walk the corpus defensively, read every file type, build and save the search index."""
    from rag import llm, readers, search, vision

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if llm.model_available():
        llm.start_server()              # start loading the model now; it overlaps with reading files
    image_reader = vision.get_image_reader()
    chunks, manifest = [], []
    for root, dirs, files in os.walk(corpus, onerror=lambda e: None):
        dirs.sort()
        for name in sorted(files):
            path = Path(root) / name
            rel = path.relative_to(corpus).as_posix()
            res = readers.read_file(path, rel, image_reader)
            chunks.extend(res.chunks)
            manifest.append({"file": rel, "status": res.status, "chunks": len(res.chunks), "note": res.note})
    idx = search.build_index(chunks, manifest)
    search.save_index(idx, INDEX_DIR)
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bad = [m for m in manifest if m["status"] not in ("ok", "image")]
    print(f"indexed {len(manifest)} files, {len(chunks)} chunks; skipped {len(bad)}", file=sys.stderr)
    for m in bad:
        print(f"  skipped {m['file']}: {m['status']} ({m['note']})", file=sys.stderr)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Retrieve candidate excerpts (BM25 plus a second hop), then let the resident model read them."""
    from rag import llm, search

    idx = search.load_index(INDEX_DIR)
    if idx is None:                           # index pass was lost: rebuild rather than fail
        index(corpus)
        idx = search.load_index(INDEX_DIR)
    if idx is None:
        return "", [], 0.0
    missing = search.unknown_ids(idx, query)
    if missing:                               # the question is about something that is not in the corpus
        print(f"question names {missing}, which appear nowhere in the corpus; refusing", file=sys.stderr)
        return "", [], 0.0
    cands = search.retrieve(idx, query)
    backend = llm.get_backend(wait=20)
    if backend is None:
        print("no model backend configured (MC3_LLM_SOCKET); refusing", file=sys.stderr)
        return "", [], 0.0
    try:
        return llm.answer_with_llm(backend, query, cands, corpus)
    except Exception as exc:                  # a model hiccup must still produce a well-formed file
        print(f"model call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return "", [], 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    # Written whole rather than streamed: a partially-written file that the
    # harness reads mid-flush parses as invalid JSON and scores the query zero.
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
