#!/usr/bin/env python3
"""Score a RUNNING container against the 10 sample questions, one fresh process per question,
exactly like the harness. Needs Docker and a GPU; run it on the build machine.

    docker run -d --name mc3test --network none --device=/dev/kfd --device=/dev/dri \\
        --group-add video --ipc=host --shm-size=16g YOURUSER/mc3-rag:v1
    docker cp mc3-corpus/. mc3test:/app/corpus
    docker exec mc3test python3 /app/app.py --index /app/corpus
    python3 score_samples.py mc3test
    python3 score_samples.py --local mc3-corpus     # no Docker: runs app.py directly (needs the env vars set)

A question scores only if the answer AND the citation set are both exactly right, as in the real grading.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def squash(s: str) -> str:
    """Like the grader: case, spaces, commas, degree signs and trailing units do not matter."""
    s = (s or "").lower().replace("\u00b0", "").strip()
    s = re.sub(r"(?<=\d)\s*[a-z%/\u00b7\u00b5\u00b2\u00b3]+$", "", s)
    return re.sub(r"[\s,]+", "", s)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = hasattr(os, "geteuid") and os.geteuid() == 0
    argv = sys.argv[1:]
    qfile = HERE / "sample-questions.json"
    if "--questions" in argv:                  # e.g. --questions hard-questions.json
        i = argv.index("--questions")
        qfile = Path(argv[i + 1])
        del argv[i:i + 2]
    local = argv[0] == "--local"              # run app.py directly, no Docker (for Colab/Kaggle tests)
    name = argv[1] if local else argv[0]
    qs = json.loads(qfile.read_text())["queries"]
    score, skipped = 0, 0
    for q in qs:
        if q.get("skip_if_root") and root:
            print(f"Q{q['n']:>2}  SKIP  (an unreadable-file question: as root the file CAN be read, so it cannot be tested here)")
            skipped += 1
            continue
        qid = f"query_{q['n']:02d}"
        t0 = time.time()
        stderr = ""
        try:
            if local:
                proc = subprocess.run([sys.executable, str(HERE / "starter" / "app" / "app.py"), "--corpus", name,
                                       "--query-id", qid, "--query", q["query"]], capture_output=True, text=True, timeout=60)
                stderr = proc.stderr
                raw = (Path(os.environ["MC2_OUTPUT_DIR"]) / f"{qid}_output.json").read_text()
            else:
                subprocess.run(["docker", "exec", name, "python3", "/app/app.py", "--corpus", "/app/corpus",
                                "--query-id", qid, "--query", q["query"]], capture_output=True, text=True, timeout=60)
                raw = subprocess.run(["docker", "exec", name, "cat", f"/app/output/{qid}_output.json"],
                                     capture_output=True, text=True, timeout=30).stdout
            out = json.loads(raw)
        except Exception as exc:
            print(f"Q{q['n']:>2}  ERROR {type(exc).__name__}: {exc}")
            continue
        secs = time.time() - t0
        accepted = {squash(q["expected_answer"])} | {squash(a) for a in q.get("answer_aliases", [])}
        ok_ans = squash(out.get("answer", "")) in accepted
        ok_cit = sorted(out.get("citations", [])) == sorted(q["expected_citations"])
        score += ok_ans and ok_cit
        flag = "PASS" if ok_ans and ok_cit else "FAIL"
        print(f"Q{q['n']:>2}  {flag}  {secs:5.1f}s  answer={out.get('answer')!r} (want {q['expected_answer']!r})  "
              f"citations={'ok' if ok_cit else out.get('citations')}")
        if stderr.strip() and not (ok_ans and ok_cit):
            print("      process said: " + stderr.strip().replace("\n", "\n      "))
        if secs > 30:
            print("      ^ over the 30 s per-question limit")
    print(f"\n{score}/{len(qs) - skipped} questions fully correct" + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
