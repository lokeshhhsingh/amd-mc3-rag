"""Tokeniser, BM25 index (stdlib only), persistence, and two-hop retrieval."""
from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

INDEX_FILE = "index.pkl"
K1, B = 1.5, 0.75

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the
this to was were what when where which who whom whose why will with does do did""".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*#?")
_SPLIT = re.compile(r"[-_.#/]")
_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b")
_SUPERSEDED = re.compile(r"\b(withdrawn|superseded|obsolete|deprecated)\b|not approved", re.I)
_DRAFT = re.compile(r"\bDRAFT\b")     # upper case only: "replaces the draft datasheet" in an approved brief is not a draft
_ASKS_ABOUT_OLD = re.compile(r"\b(withdrawn|superseded|obsolete|deprecated|draft|r1|rev(?:ision)?\s*1|old|previous|original)\b", re.I)


def _stem(tok: str) -> str:
    if tok.isalpha() and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokens(text: str) -> list:
    """Whole identifiers (tq-40, orr-1847, therm_alert#) AND their pieces are both indexed,
    so exact ids match strongly while plain words still match loosely."""
    out = []
    for m in _TOKEN.finditer(text):
        whole = m.group(0).lower()
        parts = [p for p in _SPLIT.split(whole) if p]
        if len(parts) > 1:
            out.append(whole)
        out.extend(_stem(p) for p in parts)
    return out


def query_tokens(text: str) -> list:
    return [t for t in tokens(text) if t not in STOP]


# --------------------------------------------------------------------------- #
def superseded_files(chunks_by_file: dict) -> set:
    """Soft flag: the file name or the first lines say withdrawn/superseded/obsolete."""
    flagged = set()
    for f, chunks in chunks_by_file.items():
        head = f + " " + (chunks[0].text[:250] if chunks else "")
        if _SUPERSEDED.search(head) or _DRAFT.search(head) or "draft" in f.lower():
            flagged.add(f)
    return flagged


def build_index(chunks: list, manifest: list) -> dict:
    by_file = {}
    for c in chunks:
        by_file.setdefault(c.file, []).append(c)
    flagged = superseded_files(by_file)
    docs, postings, doc_len = [], {}, []
    for c in chunks:
        did = len(docs)
        docs.append({"file": c.file, "loc": c.loc, "kind": c.kind, "text": c.text,
                     "superseded": c.file in flagged})
        toks = tokens(c.file) * 2 + tokens(c.loc) + tokens(c.text)
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, n in counts.items():
            postings.setdefault(t, []).append((did, n))
        doc_len.append(len(toks))
    n = len(docs)
    return {"docs": docs, "postings": postings, "doc_len": doc_len,
            "avgdl": (sum(doc_len) / n) if n else 0.0, "N": n, "manifest": manifest}


def save_index(idx: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / (INDEX_FILE + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(directory / INDEX_FILE)       # atomic: never leave a half-written index


def load_index(directory: Path):
    p = directory / INDEX_FILE
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def bm25(idx: dict, qtoks: list, restrict_penalty: bool) -> dict:
    scores = {}
    N, avgdl = idx["N"], idx["avgdl"] or 1.0
    for t in set(qtoks):
        plist = idx["postings"].get(t)
        if not plist:
            continue
        idf = math.log(1 + (N - len(plist) + 0.5) / (len(plist) + 0.5))
        boost = 1.6 if "-" in t or "_" in t or "#" in t or any(ch.isdigit() for ch in t) else 1.0
        for did, tf in plist:
            dl = idx["doc_len"][did]
            s = idf * boost * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
            scores[did] = scores.get(did, 0.0) + s
    if restrict_penalty:
        for did in scores:
            if idx["docs"][did]["superseded"]:
                scores[did] *= 0.6
    return scores


def top_chunks(idx: dict, query: str, k: int, per_file_cap: int = 3, exclude_files=()) -> list:
    qt = query_tokens(query)
    penalise = not _ASKS_ABOUT_OLD.search(query)
    ranked = sorted(bm25(idx, qt, penalise).items(), key=lambda kv: -kv[1])
    out, per_file = [], {}
    for did, score in ranked:
        f = idx["docs"][did]["file"]
        if f in exclude_files or per_file.get(f, 0) >= per_file_cap:
            continue
        per_file[f] = per_file.get(f, 0) + 1
        out.append((did, score))
        if len(out) >= k:
            break
    return out


def _rare_ids(idx: dict, text: str, exclude: set, max_df: int, limit: int, qset: set) -> list:
    """Identifiers (ticket ids, error codes, part numbers) in `text` worth following. Ids that occur in many chunks
    (cell names, product names) are noise. Among the rest, those whose surrounding lines share words with the
    QUESTION come first ("replace relay part BRK-REL-77" for a question about the part to replace), rarest next.
    That picks the right id out of a long list (a knowledge base of error codes)."""
    lines = text.splitlines() or [text]
    scored, seen = [], set()
    for m in _ID.findall(text):
        tok = m.lower()
        if tok in exclude or tok in seen or len(m) < 4 or not any(ch.isdigit() for ch in m):
            continue
        seen.add(tok)
        df = len(idx["postings"].get(tok, ()))
        if not 1 <= df <= max_df:
            continue
        near = set()
        for i, ln in enumerate(lines):
            if tok in ln.lower():
                near |= set(query_tokens(" ".join(lines[max(0, i - 1):i + 2])))
        near -= set(tokens(m))
        scored.append((-len(near & qset), df, len(scored), tok, m))
    scored.sort()
    return [(t[3], t[4]) for t in scored[:limit]]


def _hits_for_id(idx: dict, tok: str, src_file: str, per_id: int, qscore: dict, per_file: int = 2) -> list:
    """Chunks that contain this exact identifier, best-matching the QUESTION first (so the USD sheet beats the EUR
    sheet when the question says USD), at most `per_file` per file, never from the file that named it."""
    ranked = sorted(bm25(idx, [tok], False).items(), key=lambda kv: (-qscore.get(kv[0], 0.0), -kv[1]))
    out, files = [], {}
    for did, score in ranked:
        f = idx["docs"][did]["file"]
        if f == src_file or files.get(f, 0) >= per_file:
            continue
        files[f] = files.get(f, 0) + 1
        out.append((did, score))
        if len(out) >= per_id:
            break
    return out


def retrieve(idx: dict, query: str, k: int = 8, depth: int = 2, seeds: int = 5, ids_per_seed: int = 3,
             per_id: int = 3, level_caps: tuple = (8, 6)) -> list:
    """First pass: BM25 on the question. Then up to `depth` hops: identifiers found in the best chunks
    (that the question did not mention) are searched for directly, and ids found in THOSE chunks are
    followed once more. That is the only route to evidence whose own words the question never uses
    (log -> error knowledge base -> price list).
      * seeds are the best chunk of each of the best FILES, so one big file cannot crowd the others out;
      * each hop level has its own budget, so noise at level 1 cannot starve level 2;
      * followed chunks are ranked by how well they match the QUESTION, and each remembers which chunk led to it."""
    qscore = bm25(idx, query_tokens(query), not _ASKS_ABOUT_OLD.search(query))
    first = top_chunks(idx, query, k)
    chosen = {did: {"score": s, "via": "bm25", "src": None} for did, s in first}
    max_df = max(4, int(idx["N"] * 0.02))
    seen = set(i.lower() for i in _ID.findall(query))
    qset = set(query_tokens(query))
    frontier, files = [], set()
    for did, _ in first:
        f = idx["docs"][did]["file"]
        if f not in files:
            files.add(f)
            frontier.append(did)
        if len(frontier) >= seeds:
            break
    for level in range(depth):
        new = []
        for did in frontier:
            src_file = idx["docs"][did]["file"]
            for tok, ident in _rare_ids(idx, idx["docs"][did]["text"], seen, max_df, ids_per_seed, qset):
                seen.add(tok)
                for hdid, _ in _hits_for_id(idx, tok, src_file, per_id, qscore):
                    if hdid not in chosen and len(new) < level_caps[min(level, len(level_caps) - 1)]:
                        chosen[hdid] = {"score": qscore.get(hdid, 0.0), "via": f"hop:{ident}", "src": did}
                        new.append(hdid)
        frontier, seen_files = [], set()          # next level: the best-matching chunk of each file
        for d in sorted(new, key=lambda d: -qscore.get(d, 0.0)):
            f = idx["docs"][d]["file"]
            if f not in seen_files and len(frontier) < seeds:
                seen_files.add(f)
                frontier.append(d)
        if not frontier:
            break
    cands = []
    for did, meta in sorted(chosen.items(), key=lambda kv: (kv[1]["via"] != "bm25", -kv[1]["score"])):
        d = idx["docs"][did]
        rare = {m.lower() for m in _ID.findall(d["text"]) if any(ch.isdigit() for ch in m) and len(m) >= 5
                and 1 <= len(idx["postings"].get(m.lower(), ())) <= max_df}
        cands.append({"doc": did, "file": d["file"], "loc": d["loc"], "kind": d["kind"],
                      "text": d["text"], "superseded": d["superseded"], "rare_ids": rare,
                      "score": round(meta["score"], 3), "via": meta["via"], "src_doc": meta["src"]})
    return cands


_Q_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")


def unknown_ids(idx: dict, query: str) -> list:
    """Hyphenated identifiers (HX-15, TQ-40, HRT-1187) that the question names but that appear nowhere in the
    corpus. Asking about a product or ticket that does not exist is a question with no answer."""
    missing = []
    for m in _Q_ID.findall(query):
        low = m.lower()
        if any(ch.isdigit() for ch in m) and low not in idx["postings"] and low.replace("-", "") not in idx["postings"]:
            missing.append(m)
    return missing
