"""Prompt building, answer parsing, and the rules that turn model output into a graded answer.

The model never decides what gets cited directly. It says which numbered excerpts it needed;
we map those to files. The model is reached through a Backend, a resident process behind a
unix socket (the model is NOT loaded per question: every question is a fresh process).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from .readers import est_tokens

STRICT_GROUNDING = True   # refuse text answers that appear in none of the retrieved excerpts
MAX_EXCERPT_CHARS = 1400
MAX_CANDIDATES = 14
MAX_PER_FILE = 2             # one big noisy log must not use up the whole prompt
MAX_EXCERPT_TOKENS = 600     # no single excerpt may use more than this
# Total excerpt budget in TOKENS (not characters: digits cost one token each). 2,800 is safe on a 15 GB T4;
# the Dockerfile raises it for the big AMD GPU through the environment.
MAX_PROMPT_TOKENS = int(os.environ.get("MC3_MAX_PROMPT_TOKENS", "2800"))
MAX_IMAGES = 2

RULES = """You answer ONE question using ONLY the numbered excerpts below.

1. Use only the excerpts. Never use outside knowledge: the products and companies here are fictional.
2. "answer" is the VALUE ONLY: a number, part number, version, quarter or code. No sentence, no explanation.
   Keep qualifiers that are part of the value (for example "Q3 FY27", not "Q3"). Give the whole value.
3. An excerpt marked [WARNING: looks withdrawn/superseded/draft] is outdated or not approved. Prefer the current, approved document.
   If excerpts give different values for different years or revisions, use the newest, unless the question names a year or revision.
4. "evidence" lists the numbers of the excerpts you needed. Include one only if removing it would make the
   answer impossible. If the answer needs two (one gives an id, another gives the value for that id), list both.
   Also list any excerpt that only told you which id, code or part number to look up in another excerpt.
5. Excerpts marked [IMAGE ATTACHED] are pictures you can look at. Read the value from the picture.
6. If the excerpts do not contain the answer, reply {"answer": "", "evidence": []}. Never guess.

Reply with ONE JSON object and nothing else, for example: {"answer": "94", "evidence": [2]}"""

_REFUSALS = {"", "none", "n/a", "na", "null", "unknown", "not found", "not available",
             "not in the corpus", "i don't know", "i do not know", "no answer"}


def select_candidates(cands: list, max_tokens: int | None = None) -> list:
    """The excerpts actually shown to the model (the model's evidence numbers refer to THIS list).
    The two best direct hits always go in. The rest are taken by relevance per token, so a small precise excerpt
    (a release-notes entry) is not crowded out by large low-value ones (a spreadsheet overview) when the budget is
    tight. At most MAX_PER_FILE per file. The result keeps the original ranking order."""
    budget = max_tokens or MAX_PROMPT_TOKENS
    cost = lambda c: min(est_tokens(c["text"]), MAX_EXCERPT_TOKENS) + 12
    value = lambda c: (c.get("score", 0.0) + 0.01) / (cost(c) ** 0.5)
    chosen, used, per_file = [], 0.0, {}

    def take(c) -> bool:
        nonlocal used
        direct_hit = c.get("via", "bm25") == "bm25"
        if any(c is x for x in chosen) or len(chosen) >= MAX_CANDIDATES or (direct_hit and per_file.get(c["file"], 0) >= MAX_PER_FILE):
            return False
        if chosen and used + cost(c) > budget:
            return False
        chosen.append(c)
        used += cost(c)
        per_file[c["file"]] = per_file.get(c["file"], 0) + 1
        return True

    direct = [c for c in cands if c.get("via", "bm25") == "bm25"]
    for c in direct[:2]:
        take(c)
    for c in sorted([c for c in cands if not any(c is x for x in chosen)], key=lambda c: -value(c)):
        take(c)
    return [c for c in cands if any(c is x for x in chosen)]


def _clip(text: str) -> str:
    t = est_tokens(text)
    return text[:MAX_EXCERPT_CHARS] if t <= MAX_EXCERPT_TOKENS else text[:int(len(text) * MAX_EXCERPT_TOKENS / t)]


def build_prompt(question: str, cands: list) -> str:
    parts = [RULES, "", f"QUESTION: {question}", "", "EXCERPTS:"]
    for i, c in enumerate(cands, start=1):
        tag = f"[{i}] {c['file']} ({c['loc']})"
        if c["superseded"]:
            tag += " [WARNING: looks withdrawn/superseded/draft]"
        if c["kind"] == "image":
            tag += " [IMAGE ATTACHED]"
        parts.append(f"{tag}\n{_clip(c['text'])}\n")
    parts.append('JSON:')
    return "\n".join(parts)


def parse_response(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ev = obj.get("evidence", [])
            ev = [int(x) for x in ev] if isinstance(ev, list) else []
            return {"answer": str(obj.get("answer", "")), "evidence": ev}
        except Exception:
            pass
    a = re.search(r'"answer"\s*:\s*"([^"]*)"', raw or "")
    return {"answer": a.group(1) if a else "", "evidence": []}


def normalise_answer(ans: str) -> str:
    a = re.sub(r"\s+", " ", (ans or "")).strip()
    a = re.sub(r"^(answer|value)\s*:\s*", "", a, flags=re.I)
    while True:                               # peel quotes/backticks and a sentence-final period, in any order
        b = a.strip(" `\"'")
        if b.endswith(".") and not re.search(r"\d\.$", b):
            b = b[:-1]
        if b == a:
            break
        a = b
    if re.fullmatch(r"(?:\u00b1|\+/-)\s*\d[\d.,]*\s*[A-Za-z%/\u00b7\u00b5\u00b2\u00b3 ]*", a):
        a = re.sub(r"^(?:\u00b1|\+/-)\s*", "", a)          # "+/-0.04 mm" -> "0.04 mm"
    return "" if a.lower() in _REFUSALS else a


def _norm_text(s: str) -> str:
    s = s.lower().replace("\u00b1", " ").replace("+/-", " ").replace("\u00b0", "")
    return re.sub(r"\s+", " ", s)


def _find(needle: str, hay: str) -> bool:
    """needle as a standalone token: '15' is not in '115', '5' is not in '5.2.1', '15' is not in '2026-08-15'."""
    if not needle:
        return False
    return re.search(r"(?<![a-z0-9.\-:/])" + re.escape(needle) + r"(?![a-z0-9]|[.\-:/]\d)", hay) is not None


def contains_answer(ans: str, text: str) -> bool:
    a, t = _norm_text(ans).strip(), _norm_text(text)
    if _find(a, t):
        return True
    m = re.fullmatch(r"(-?\d[\d.,]*)\s*[a-z%/\u00b7\u00b5\u00b2\u00b3 ]*", a)   # a number with a unit: try the number alone
    return bool(m) and _find(m.group(1), t)


_QUESTION_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")
_LINK_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b|\b[A-Z]\d{4,}\b")


def _linked(a: str, b: str) -> bool:
    """Do two excerpts share a specific identifier (ticket id, error code, part number)? Then one explains the other."""
    ids = lambda t: {m.lower() for m in _LINK_ID.findall(t) if len(m) >= 5 and any(ch.isdigit() for ch in m)
                     and not re.fullmatch(r"[A-Z]\d-\d+", m)}
    return bool(ids(a) & ids(b))


def _ids_of(text: str) -> set:
    return {m.lower() for m in _LINK_ID.findall(text) if len(m) >= 5 and any(ch.isdigit() for ch in m)
            and not re.fullmatch(r"[A-Z]\d-\d+", m)}


def _rare(c: dict) -> set:
    """Specific ids in an excerpt that occur in only a few chunks of the corpus (retrieval tags them)."""
    return c["rare_ids"] if "rare_ids" in c else _ids_of(c["text"])


def _near(text: str, is_anchor, ident: str, window: int = 1) -> bool:
    """Is `ident` on (or within `window` lines of) a line that satisfies `is_anchor` (the answer, or a link id)?"""
    lines = text.splitlines() or [text]
    anchors = [i for i, ln in enumerate(lines) if is_anchor(ln)]
    at_id = [i for i, ln in enumerate(lines) if ident in ln.lower()]
    return any(abs(a - b) <= window for a in anchors for b in at_id)


def _walk_links(cands: list, keep: set, ans: str, skip: str) -> set:
    """Follow the chain that led to the answer. From an excerpt that holds the answer (or from one already linked),
    another excerpt is a LINK if it holds no answer itself, shares exactly ONE rare id with it, that id is neither in
    the question nor in the answer, has not been used already, and sits right next to the anchor (the answer, or the
    id that led here). Several shared ids mean two lists that merely overlap; a common id is no link at all.
    The log that names an error code, the knowledge-base entry that names the part: each is a link. A chain of links
    is kept only if it ends at an excerpt the question's own words retrieved; a dead end is just related reading."""
    base = set(keep)
    todo = [(k, lambda ln, a=ans: contains_answer(a, ln)) for k in base]
    used, parent = set(), {}
    while todo:
        k, is_anchor = todo.pop()
        for i, c in enumerate(cands, start=1):
            if i in base or i in parent or contains_answer(ans, c["text"]):
                continue
            shared = [x for x in _rare(c) & _rare(cands[k - 1]) if x not in skip and x not in used]
            if len(shared) != 1 or not _near(cands[k - 1]["text"], is_anchor, shared[0]):
                continue
            parent[i] = k
            used.add(shared[0])
            todo.append((i, lambda ln, x=shared[0]: x in ln.lower()))
    children = {}
    for i, k in parent.items():
        children.setdefault(k, []).append(i)

    def reaches_origin(i: int) -> bool:       # a link counts only if the chain leads to an excerpt the QUESTION pointed at
        return cands[i - 1].get("via", "bm25") == "bm25" or any(reaches_origin(j) for j in children.get(i, []))
    return base | {i for i in parent if reaches_origin(i)}


def finalize(parsed: dict, cands: list, question: str = ""):
    """-> (answer, citations, confidence). Refusal is ('', [], 0.0).
    The model's evidence list is verified, not trusted: a cited excerpt must contain the answer, or share an
    identifier with one that does (the excerpt that named the ticket or code that led to the answer)."""
    ans = normalise_answer(parsed.get("answer", ""))
    if not ans:
        return "", [], 0.0
    cited = sorted({i for i in parsed.get("evidence", []) if 1 <= i <= len(cands)})
    holders = [i for i, c in enumerate(cands, start=1) if c["kind"] == "text" and contains_answer(ans, c["text"])]
    has_image = any(c["kind"] == "image" for c in cands)
    if STRICT_GROUNDING and not holders and not has_image:
        return "", [], 0.0                    # nothing retrieved contains it: likely recalled, so refuse
    keep = {i for i in cited if i in holders or cands[i - 1]["kind"] == "image"}
    if not keep and holders:
        keep = {holders[0]}                   # the model cited nothing useful: use the best excerpt that has the answer
    if not keep:
        return "", [], 0.0                    # cannot cite, so cannot answer
    qids = {m.lower() for m in _QUESTION_ID.findall(question) if any(ch.isdigit() for ch in m)}
    if qids:                                  # the question names a ticket/product/code: cite the excerpt that has it,
        names = lambda i: all(q in _norm_text(cands[i - 1]["text"]) for q in qids)
        strong = {i for i in keep if names(i)}
        if strong:                            # not another file that merely contains the same value
            keep = strong
        else:                                 # the cited excerpts never mention it: use the holder that does, if there is one
            named = [i for i in holders if names(i)]
            if named:
                keep = set(named[:2])
    for i in cited:                           # keep cited excerpts that explain how the answer was reached
        if i not in keep and any(_linked(cands[i - 1]["text"], cands[k - 1]["text"]) for k in keep):
            keep.add(i)
    keep = _walk_links(cands, keep, ans, (question + " " + ans).lower())
    files = sorted({cands[i - 1]["file"] for i in keep})
    return ans, files, 0.9 if holders else 0.5


# --------------------------------------------------------------------------- #
SOCK_DEFAULT = "/tmp/mc3_model.sock"
TRANSCRIBE_PROMPT = ("Transcribe all text visible in this image exactly as written, keeping part numbers, "
                     "pin names, revisions and codes. Then add one sentence saying what the image shows "
                     "(diagram, label, photo, table). Output plain text only.")


def sock_path() -> str:
    return os.environ.get("MC3_LLM_SOCKET", SOCK_DEFAULT)


def model_dir() -> Path:
    return Path(os.environ.get("MC3_MODEL_DIR", "/models/qwen3-vl-8b"))


class SocketBackend:
    """Client for the resident model process. Protocol: one JSON line each way."""

    def __init__(self, path: str, timeout: float = 22.0):
        self.path, self.timeout = path, timeout

    def _round_trip(self, payload: dict, timeout: float) -> dict:
        import socket
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(self.path)
            s.sendall((json.dumps(payload) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                part = s.recv(65536)
                if not part:
                    break
                buf += part
        return json.loads(buf.decode())

    def complete(self, prompt: str, images: list, max_new_tokens: int = 160) -> str:
        reply = self._round_trip({"prompt": prompt, "images": [str(p) for p in images],
                                  "max_new_tokens": max_new_tokens}, self.timeout)
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply["text"]

    def ping(self) -> bool:
        try:
            return self._round_trip({"ping": 1}, 2.0).get("text") == "pong"
        except Exception:
            return False


def ping(path: str) -> bool:
    return SocketBackend(path).ping()


def _server_pid_alive(path: str) -> bool:
    """Is the server process running? A dead child that nobody has reaped yet is a zombie, and
    os.kill(pid, 0) still succeeds on a zombie, so read its state from /proc as well."""
    try:
        pid = int(Path(path + ".pid").read_text())
        os.kill(pid, 0)
    except Exception:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except Exception:
        return True


def model_available() -> bool:
    """True if a model server is running or can be started from the shipped weights."""
    return (ping(sock_path()) or model_dir().is_dir() or bool(os.environ.get("MC3_FAKE_MODEL")))


def start_server() -> bool:
    """Start the resident model process, fully detached, so it outlives this `docker exec`.
    Its stdio goes to a log file: a child that keeps the exec's pipes open makes `docker exec`
    hang until it exits. Does nothing if one is already running or loading (one copy only:
    two would double the VRAM)."""
    import subprocess
    path = sock_path()
    if ping(path) or _server_pid_alive(path):
        return False
    if not (model_dir().is_dir() or os.environ.get("MC3_FAKE_MODEL")):
        return False
    script = Path(__file__).resolve().parent.parent / "model_server.py"
    with open("/tmp/mc3_model_server.log", "ab") as logf:
        subprocess.Popen([sys.executable, str(script)], stdin=subprocess.DEVNULL, stdout=logf,
                         stderr=logf, start_new_session=True, close_fds=True,
                         env=dict(os.environ, MC3_LLM_SOCKET=path))
    return True


def wait_ready(timeout: float) -> bool:
    """Block until the model answers a ping. Gives up early if the server process died
    (for example: no GPU was visible), instead of waiting out the whole timeout."""
    path, t0 = sock_path(), time.time()
    while time.time() - t0 < timeout:
        if ping(path):
            return True
        if not _server_pid_alive(path) and time.time() - t0 > 3:
            return False
        time.sleep(0.5)
    return False


def get_backend(wait: float = 0.0):
    """A ready backend, or None. With wait > 0, start the server if needed and wait for it."""
    path = sock_path()
    if ping(path):
        return SocketBackend(path)
    if wait > 0 and model_available():
        start_server()
        if wait_ready(wait):
            return SocketBackend(path)
    return None


_ENTITY = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b")


def _question_entities(question: str) -> list:
    """Codes and model numbers the question asks ABOUT (E101, HX-12, HRT-1187): letters and digits, 3+ chars."""
    return [m for m in dict.fromkeys(_ENTITY.findall(question))
            if len(m) >= 3 and any(ch.isdigit() for ch in m) and any(ch.isalpha() for ch in m)]


def _mentions(c: dict, ents: list) -> bool:
    toks = set(re.split(r"[^a-z0-9]+", (c["file"] + " " + c["text"]).lower()))
    return all(e.lower() in toks or e.lower().replace("-", "") in toks for e in ents)


def _ask(backend, question: str, shown: list, corpus: Path) -> tuple:
    images = [corpus / c["file"] for c in shown if c["kind"] == "image"][:MAX_IMAGES]
    raw = backend.complete(build_prompt(question, shown), images, max_new_tokens=120)
    if os.environ.get("MC3_DEBUG"):
        print(f"RAW MODEL REPLY: {raw!r}", file=sys.stderr)
    return finalize(parse_response(raw), shown, question)


def answer_with_llm(backend, question: str, cands: list, corpus: Path):
    budget = MAX_PROMPT_TOKENS
    for attempt in range(3):
        shown = select_candidates(cands, budget)
        try:
            result = _ask(backend, question, shown, corpus)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or attempt == 2:
                raise
            budget = int(budget * 0.5)         # a small GPU: try again with half as many excerpts
    ans, files, _ = result
    ents = _question_entities(question)
    if ans and ents and not any(c["kind"] == "image" for c in shown):
        carriers = [c for c in shown if _mentions(c, ents)]
        answered_from = [c for c in shown if c["file"] in files]
        if carriers and not any(_mentions(c, ents) for c in answered_from):
            # The answer came from excerpts that never mention what was asked about (an answer about E102 given for
            # E101). Ask once more, showing only the excerpts that do mention it. No answer there means no answer.
            try:
                return _ask(backend, question, carriers[:6], corpus)
            except RuntimeError:
                return result
    return result
