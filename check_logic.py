#!/usr/bin/env python3
"""Offline checks of the answer rules and of the app.py contract, using a FAKE model.

    python3 check_logic.py mc3-corpus

Part 1 tests parse/normalise/grounding/citation rules on canned model replies.
Part 2 runs app.py exactly the way the harness does (--index, then one process per
question) against a fake model on a unix socket, and checks the JSON files it writes.
The fake model is not intelligent: it only proves the plumbing is right.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "starter" / "app"
sys.path.insert(0, str(APP))
from rag import llm  # noqa: E402

cands = [
    {"file": "specs/tq40_datasheet_r2.pdf", "loc": "page 1", "kind": "text", "superseded": False,
     "text": "Maximum junction temperature .......... 94 C"},
    {"file": "support/bug_database.csv", "loc": "row 2", "kind": "text", "superseded": False,
     "text": "ticket=ORR-1847; fixed_in=4.3.2"},
    {"file": "logs/prod.log", "loc": "lines 1-5", "kind": "text", "superseded": False,
     "text": "ERROR E7731: thermal throttle engaged ... incident logged against ORR-1847"},
]

def fin(raw):
    return llm.finalize(llm.parse_response(raw), cands)

# ---- part 1: rules ------------------------------------------------------- #
assert fin('{"answer": "94", "evidence": [1]}') == ("94", ["specs/tq40_datasheet_r2.pdf"], 0.9)
assert fin('Sure! {"answer": "4.3.2", "evidence": [3, 2]}')[1] == ["logs/prod.log", "support/bug_database.csv"]  # two-file answer
assert fin('{"answer": "", "evidence": []}') == ("", [], 0.0)                       # honest refusal
assert fin('{"answer": "I don\'t know", "evidence": [1]}') == ("", [], 0.0)         # prose refusal -> empty
assert fin('{"answer": "6412", "evidence": [1]}') == ("", [], 0.0)                  # not in any excerpt -> recalled, refuse
assert fin('{"answer": "94", "evidence": []}')[1] == ["specs/tq40_datasheet_r2.pdf"]  # missing evidence: cite where it was found
assert fin('{"answer": "94", "evidence": [9]}')[1] == ["specs/tq40_datasheet_r2.pdf"]  # bogus evidence ids ignored
assert fin("total garbage") == ("", [], 0.0)                                         # unparseable -> well-formed refusal
assert llm.normalise_answer('Answer: "E7731".') == "E7731"
assert llm.normalise_answer("4.3.2") == "4.3.2"                                      # trailing dots in versions survive
hop_cands = [
    {"doc": 5, "file": "logs/prod.log", "loc": "l1", "kind": "text", "superseded": False, "via": "bm25", "src_doc": None,
     "text": "ERROR E7731 thermal throttle, ticket ORR-1847"},
    {"doc": 9, "file": "support/bug_database.csv", "loc": "row 2", "kind": "text", "superseded": False, "via": "hop:ORR-1847", "src_doc": 5,
     "text": "ticket=ORR-1847; fixed_in=4.3.2"},
    {"doc": 7, "file": "specs/other.pdf", "loc": "p1", "kind": "text", "superseded": False, "via": "bm25", "src_doc": None, "text": "unrelated"},
]
r = llm.finalize(llm.parse_response('{"answer": "4.3.2", "evidence": [2]}'), hop_cands)
assert r[1] == ["logs/prod.log", "support/bug_database.csv"], r        # the hop's source is cited too
r = llm.finalize(llm.parse_response('{"answer": "4.3.2", "evidence": [1, 2]}'), hop_cands)
assert r[1] == ["logs/prod.log", "support/bug_database.csv"], r        # no duplicates when the model already listed both
r = llm.finalize(llm.parse_response('{"answer": "unrelated", "evidence": [3]}'), hop_cands)
assert r[1] == ["specs/other.pdf"], r                                    # an ordinary hit is untouched
chain = [
    {"doc": 5, "file": "logs/prod.log", "loc": "l", "kind": "text", "superseded": False, "via": "bm25", "src_doc": None, "text": "STOP code=HX-ERR-4471"},
    {"doc": 6, "file": "kb/errors.md", "loc": "s", "kind": "text", "superseded": False, "via": "hop:HX-ERR-4471", "src_doc": 5, "text": "HX-ERR-4471: replace BRK-REL-77"},
    {"doc": 9, "file": "finance/prices.xlsx", "loc": "r", "kind": "text", "superseded": False, "via": "hop:BRK-REL-77", "src_doc": 6, "text": "part=BRK-REL-77; price=38.50"},
]
r = llm.finalize(llm.parse_response('{"answer": "38.50", "evidence": [3]}'), chain)
assert r[1] == ["finance/prices.xlsx", "kb/errors.md", "logs/prod.log"], r   # two-hop chain: all three files are cited
from rag import readers
assert readers._format_number("38.5", "0.00") == "38.50"
assert readers._format_number("1234.5", "#,##0.00") == "1,234.50"
assert readers._format_number("0.075", "0.0%") == "7.5%"
assert readers._format_number("12", "General") == "12"
assert readers._format_number("46000", "yyyy-mm-dd") == "2025-12-09", readers._format_number("46000", "yyyy-mm-dd")
# --- the failures the real model run exposed (Colab T4, 22/31) ---
def mk(doc, file, text, **kw):
    d = {"doc": doc, "file": file, "loc": "l", "kind": "text", "superseded": False, "via": "bm25", "src_doc": None, "text": text}
    d.update(kw)
    return d
wrong = [mk(1, "fw/release_notes.txt", "5.1.0: New admin session timeout setting."),
         mk(2, "manuals/controller.docx", "Admin sessions expire after 15 minutes of inactivity."),
         mk(3, "tickets.csv", "ticket=HRT-1115; component=fan")]
assert llm.finalize(llm.parse_response('{"answer": "15", "evidence": [1]}'), wrong)[1] == ["manuals/controller.docx"]   # model cited the wrong file
assert llm.contains_answer("15", "row 115 and 2026-08-15 and 15.5") is False                                          # standalone tokens only
assert llm.contains_answer("5.2.1", "fixed in 5.2.1 today") and not llm.contains_answer("5", "version 5.2.1")
pm = [mk(1, "specs/hx12.pdf", "Repeatability ........ +/-0.04 mm")]
assert llm.finalize(llm.parse_response('{"answer": "\u00b10.04 mm", "evidence": [1]}'), pm) == ("0.04 mm", ["specs/hx12.pdf"], 0.9)   # sign dropped, still grounded
assert llm.normalise_answer("+/-0.5") == "0.5" and llm.normalise_answer("-20") == "-20"
from rag.readers import est_tokens
dense = [mk(i, f"f{i}.txt", "2026-08-14 03:12:09 " * 40) for i in range(30)]          # digit-dense: about 1 token per character
shown = llm.select_candidates(dense, 2800)
assert 0 < len(shown) <= llm.MAX_CANDIDATES
assert sum(min(est_tokens(x["text"]), llm.MAX_EXCERPT_TOKENS) + 12 for x in shown) <= 2800
direct = [mk(i, f"d{i}.txt", "word " * 150) for i in range(10)]
hop = mk(99, "kb/errors.md", "word " * 150, via="hop:HX-ERR-4471", src_doc=0, score=50.0)
assert hop in llm.select_candidates(direct + [hop], 1200)                               # a hop excerpt survives a tight budget
pref = [mk(1, "tickets.csv", "ticket=HRT-1187; fixed_in=5.2.1"), mk(2, "release_notes.txt", "Firmware 5.2.1 (2026-07-30): fixes things")]
q = "In which firmware release was ticket HRT-1187 fixed?"
assert llm.finalize(llm.parse_response('{"answer": "5.2.1", "evidence": [1, 2]}'), pref, q)[1] == ["tickets.csv"]   # the file that has the named ticket
assert llm.finalize(llm.parse_response('{"answer": "5.2.1", "evidence": [1, 2]}'), pref)[1] == ["release_notes.txt", "tickets.csv"]  # no id in the question: both held it

class OomOnce:
    calls = 0
    def complete(self, prompt, images, max_new_tokens=160):
        OomOnce.calls += 1
        if OomOnce.calls == 1:
            raise RuntimeError("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
        return '{"answer": "94", "evidence": [1]}'
r = llm.answer_with_llm(OomOnce(), "q", [mk(1, "specs/a.pdf", "Maximum junction temperature 94 C")], Path("."))
assert r[0] == "94" and OomOnce.calls == 2, (r, OomOnce.calls)                          # one out-of-memory error: retried, not refused
from rag import search
fake_idx = {"postings": {"hx-12": [(0, 1)], "hx9": [(0, 1)], "tq-40": [(0, 1)]}}
assert search.unknown_ids(fake_idx, "What is the maximum payload of the HX-15?") == ["HX-15"]     # no such product
assert search.unknown_ids(fake_idx, "What is the payload of the HX-12?") == []
assert search.unknown_ids(fake_idx, "What is the payload of the HX-9?") == []                       # HX9 spelled without the hyphen still counts
assert search.unknown_ids(fake_idx, "Which Q3 FY27 quarter? Which pin of X7?") == []              # plain words and short ids are not checked
assert llm.contains_answer("15", "ts 03:15:09 and 2026-08-15 and 115") is False and llm.contains_answer("15", "expire after 15 minutes")
bridge = [mk(1, "logs/prod.log", "STOP code=HX-ERR-4471 line4 2026-08-14"), mk(2, "fw/release_notes.txt", "5.2.1 fixes HX-ERR-4471 brake release"),
          mk(3, "tickets.csv", "ticket=HRT-9; unrelated")]
r = llm.finalize(llm.parse_response('{"answer": "5.2.1", "evidence": [2]}'), bridge, "Which release fixed the error that stopped line 4 on 2026-08-14?")
assert r[1] == ["fw/release_notes.txt", "logs/prod.log"], r                                 # the log named the code the question did not
r = llm.finalize(llm.parse_response('{"answer": "5.2.1", "evidence": [2]}'), bridge, "Which release fixed HX-ERR-4471?")
assert r[1] == ["fw/release_notes.txt"], r                                                    # the question named the code itself: no bridge
class WrongEntity:
    """Answers about E102 first; once shown only E101 excerpts, answers correctly."""
    def complete(self, prompt, images, max_new_tokens=160):
        return '{"answer": "Encoder fault", "evidence": [1]}' if "E101" in prompt and "E102" not in prompt else \
               '{"answer": "Over-temperature", "evidence": [2]}'
ents = [mk(1, "manual.docx", "E101 | Encoder fault | Recalibrate", via="bm25"), mk(2, "manual.docx", "E102 | Over-temperature", via="bm25")]
ents[0]["text"] = "E101 | Encoder fault | Recalibrate"
wrong_first = [mk(1, "kb/errors.md", "HX-ERR-4410 Encoder drift on joint J2 Over-temperature"), ents[0], ents[1]]
assert llm._question_entities("What does error E101 mean?") == ["E101"]
assert llm._question_entities("Which pin of connector X7 carries BRAKE_REL?") == []
common = [mk(1, "support/asset_label.jpg", "MODEL TQ-40\nBOARD REVISION REV-C2", kind="image", rare_ids=set()),
          mk(2, "specs/datasheet.pdf", "TQ-40 datasheet, board REV-C2 shown on the label", rare_ids=set())]
r = llm.finalize(llm.parse_response('{"answer": "REV-C2", "evidence": [1]}'), common, "What board revision is printed on the asset label?")
assert r[1] == ["support/asset_label.jpg"], r                                              # a common id (TQ-40) is no link
sibling = [mk(1, "finance/price_list_2026.xlsx", "part=BRK-REL-77; price=38.50"), mk(2, "kb/errors.md", "HX-ERR-4471: replace BRK-REL-77"),
           mk(3, "finance/price_list_2025.csv", "part=BRK-REL-77; price=34.00"), mk(4, "logs/prod.log", "STOP code=HX-ERR-4471")]
r = llm.finalize(llm.parse_response('{"answer": "38.50", "evidence": [1]}'), sibling, "What is the USD price of the part to replace for the line 4 stop?")
assert r[1] == ["finance/price_list_2026.xlsx", "kb/errors.md", "logs/prod.log"], r          # price <- kb <- log; the 2025 sibling is not a link
dead = [mk(1, "logs/prod.log", "STOP code=HX-ERR-4471 fw=5.2.0"), mk(2, "kb/errors.md", "HX-ERR-4471: brake relay failure", via="hop:HX-ERR-4471")]
r = llm.finalize(llm.parse_response('{"answer": "5.2.0", "evidence": [1]}'), dead, "Which firmware version was line 4 running at the stop?")
assert r[1] == ["logs/prod.log"], r                                                         # a dead-end explanation is not cited
from types import SimpleNamespace as NS
flagged = search.superseded_files({"specs/hx9L_datasheet_DRAFT.pdf": [NS(text="HX-9L datasheet (DRAFT)\nTargets only")],
                                   "specs/hx9L_product_brief.docx": [NS(text="Status: approved. This brief replaces the HX-9L draft datasheet.")],
                                   "specs/hx9_rev2.pdf": [NS(text="HX-9 datasheet, revision 2. Supersedes revision 1.")]})
assert flagged == {"specs/hx9L_datasheet_DRAFT.pdf"}, flagged                              # the draft yes; the brief that mentions a draft, no
named = [mk(1, "support/tickets.csv", "ticket=HRT-1187; fixed_in=5.2.1"), mk(2, "fw/release_notes.txt", "Firmware 5.2.1: Fixes HX-ERR-4471 (brake release)")]
r = llm.finalize(llm.parse_response('{"answer": "5.2.1", "evidence": [1]}'), named, "Which firmware release fixed HX-ERR-4471?")
assert r[1] == ["fw/release_notes.txt"], r                                                   # cited a row that never mentions the code: use the excerpt that does
print("part 1 ok: answer rules behave")

# ---- part 2: the real app.py, a fake model, one process per question ------ #
def fake_model(sock_path):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(4)
    def loop():
        while True:
            conn, _ = srv.accept()
            req = json.loads(conn.makefile().readline())
            if req.get("ping"):
                conn.sendall((json.dumps({"text": "pong"}) + "\n").encode()); conn.close(); continue
            p = req["prompt"]
            qlines = [l for l in p.splitlines() if l.startswith("QUESTION:")]
            if not qlines:                      # an image-transcription request made during --index
                conn.sendall((json.dumps({"text": "MOCK TRANSCRIPTION"}) + "\n").encode()); conn.close(); continue
            question = qlines[0]
            if "junction temperature" in question:
                k = next(l for l in p.splitlines() if "tq40_datasheet_r2.pdf" in l and l.startswith("["))
                reply = {"answer": "94", "evidence": [int(k[1:k.index("]")])]}
            else:
                reply = {"answer": "", "evidence": []}
            conn.sendall((json.dumps({"text": json.dumps(reply)}) + "\n").encode()); conn.close()
    threading.Thread(target=loop, daemon=True).start()

corpus = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "mc3-corpus").resolve()
tmp = Path(tempfile.mkdtemp(prefix="mc3chk-"))
sock = str(tmp / "m.sock"); fake_model(sock)
env = dict(os.environ, MC3_INDEX_DIR=str(tmp / "index"), MC2_OUTPUT_DIR=str(tmp / "out"), MC3_LLM_SOCKET=sock)
run = lambda *a: subprocess.run([sys.executable, str(APP / "app.py"), *a], env=env, capture_output=True, text=True, timeout=60)

r = run("--index", str(corpus)); assert r.returncode == 0, r.stderr
for qid, q, want in [("query_01", "What is the maximum junction temperature of the TQ-40?", ("94", ["specs/tq40_datasheet_r2.pdf"])),
                     ("query_10", "What is the unit price of the TQ-40 at 10,000 unit volume?", ("", []))]:
    r = run("--corpus", str(corpus), "--query-id", qid, "--query", q); assert r.returncode == 0, r.stderr
    out = json.loads((tmp / "out" / f"{qid}_output.json").read_text())
    assert set(out) == {"answer", "citations", "confidence"}, out
    assert (out["answer"], out["citations"]) == want, out
    print(f"part 2 ok: {qid} -> {out}")


# ---- part 3: the resident server lifecycle (fake model, no GPU needed) ----- #
import pickle, signal, time

def kill_server(sock_file):
    try:
        os.kill(int(Path(sock_file + ".pid").read_text()), signal.SIGTERM)
    except Exception:
        pass

tmp3 = Path(tempfile.mkdtemp(prefix="mc3srv-"))
sock3 = str(tmp3 / "fake.sock")
env3 = dict(os.environ, MC3_INDEX_DIR=str(tmp3 / "index"), MC2_OUTPUT_DIR=str(tmp3 / "out"),
            MC3_LLM_SOCKET=sock3, MC3_FAKE_MODEL="1", MC3_FAKE_LOAD_SECONDS="3",
            MC3_MODEL_DIR=str(tmp3 / "no-such-dir"))
run3 = lambda *a, e=env3, t=120: subprocess.run([sys.executable, str(APP / "app.py"), *a], env=e,
                                                capture_output=True, text=True, timeout=t)
try:
    t0 = time.time()
    r = run3("--index", str(corpus)); assert r.returncode == 0, r.stderr      # returns: no hang on inherited pipes
    idx = pickle.loads((tmp3 / "index" / "index.pkl").read_bytes())
    img_texts = [d["text"] for d in idx["docs"] if d["kind"] == "image"]
    assert img_texts and all("FAKE TRANSCRIPTION" in t for t in img_texts), img_texts
    print(f"part 3 ok: --index started the model server detached, waited for it ({time.time() - t0:.1f}s), "
          f"and the images were read through it")
    t0 = time.time()
    r = run3("--corpus", str(corpus), "--query-id", "query_01", "--query", "What is the maximum junction temperature of the TQ-40?")
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 10, "a question must not reload the model"
    out = json.loads((tmp3 / "out" / "query_01_output.json").read_text())
    assert set(out) == {"answer", "citations", "confidence"}
    print(f"part 3 ok: a fresh process per question reused the running server ({time.time() - t0:.1f}s)")
finally:
    kill_server(sock3)

# ---- part 4: no usable GPU / model -> give up fast, never hang -------------- #
tmp4 = Path(tempfile.mkdtemp(prefix="mc3nogpu-")); (tmp4 / "weights").mkdir()
sock4 = str(tmp4 / "x.sock")
env4 = dict(os.environ, MC3_INDEX_DIR=str(tmp4 / "index"), MC2_OUTPUT_DIR=str(tmp4 / "out"),
            MC3_LLM_SOCKET=sock4, MC3_MODEL_DIR=str(tmp4 / "weights"))
env4.pop("MC3_FAKE_MODEL", None)
try:
    t0 = time.time()
    r = run3("--index", str(corpus), e=env4, t=200); assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 60, f"gave up too slowly: {time.time() - t0:.0f}s"
    r = run3("--corpus", str(corpus), "--query-id", "query_01", "--query", "What is the maximum junction temperature?", e=env4)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp4 / "out" / "query_01_output.json").read_text())
    assert out["answer"] == "" and out["citations"] == [], out
    print(f"part 4 ok: with a server that cannot start, index and query both finish quickly ({time.time() - t0:.1f}s) with a refusal")
finally:
    kill_server(sock4)
