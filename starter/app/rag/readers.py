"""File readers for the MC3 corpus.

THE RULE: read_file() never raises. One bad file (unreadable, encrypted, binary,
corrupt, empty) must not cost you the rest of the corpus, so every failure is
caught per file and reported as a status instead.

Only the standard library is needed for txt/log/py/csv/docx/xlsx/pptx. PDFs use
pypdf (pure Python). Images are handed to an optional `image_reader` callable
(OCR or a vision model) that is supplied by the caller.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

MAX_READ_BYTES = 25 * 1024 * 1024
CHUNK_CHARS = 1200
MAX_CHUNK_TOKENS = 350          # Qwen-style tokenisers read every digit as one token, so logs are token-dense


def est_tokens(text: str) -> float:
    """Rough token count: one per digit, plus about one per 3.5 other characters."""
    digits = sum(ch.isdigit() for ch in text)
    return digits + (len(text) - digits) / 3.5
OVERLAP_LINES = 3
CSV_OVERVIEW_ROWS = 12

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
CSV_EXT = {".csv": ",", ".tsv": "\t"}

csv.field_size_limit(10_000_000)


@dataclass
class Chunk:
    file: str          # corpus-relative path, forward slashes
    loc: str           # where in the file, e.g. "page 1" or "sheet RMA Parts, row 3"
    text: str
    kind: str = "text"  # "text" or "image"


@dataclass
class ReadResult:
    file: str
    status: str         # ok | empty | unreadable | encrypted | binary | image | scanned
    chunks: list = field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------- #
def read_file(path: Path, rel: str, image_reader=None) -> ReadResult:
    try:
        return _read_file(path, rel, image_reader)
    except Exception as exc:  # per-file catch-all: this is the whole point
        return ReadResult(rel, "unreadable", [], f"{type(exc).__name__}: {exc}")


def _read_file(path: Path, rel: str, image_reader) -> ReadResult:
    ext = path.suffix.lower()
    if path.stat().st_size == 0:
        return ReadResult(rel, "empty", [], "zero bytes")
    if ext in IMAGE_EXT:
        return _read_image(path, rel, image_reader)
    if ext == ".pdf":
        return _read_pdf(path, rel)
    if ext == ".docx":
        return _read_docx(path, rel)
    if ext in (".xlsx", ".xlsm"):
        return _read_xlsx(path, rel)
    if ext == ".pptx":
        return _read_pptx(path, rel)
    if ext in CSV_EXT:
        return _read_csv(path, rel, CSV_EXT[ext])

    data = _read_bytes(path)          # PermissionError surfaces here -> "unreadable"
    if _looks_binary(data):
        return ReadResult(rel, "binary", [], "unknown binary type, skipped")
    return ReadResult(rel, "ok", _chunk_lines(rel, _decode(data), "lines"))


# --------------------------------------------------------------------------- #
def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(MAX_READ_BYTES)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeError:
            pass
    return data.decode("latin-1", errors="replace")


def _looks_binary(data: bytes) -> bool:
    head = data[:4096]
    if not head:
        return False
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):   # UTF-16 BOM: text
        return False
    if b"\x00" in head:
        return True
    printable = sum(1 for b in head if b in (9, 10, 13) or 32 <= b < 127 or b >= 128)
    return printable / len(head) < 0.85


def _chunk_lines(rel: str, text: str, label: str) -> list:
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = []
    for ln in raw:                                    # split absurdly long lines
        if len(ln) > CHUNK_CHARS:
            lines.extend(ln[i:i + CHUNK_CHARS // 2] for i in range(0, len(ln), CHUNK_CHARS // 2))
        else:
            lines.append(ln)
    chunks, start, n = [], 0, len(lines)
    while start < n:
        size, toks, end = 0, 0.0, start
        while end < n and (end == start or (size + len(lines[end]) + 1 <= CHUNK_CHARS
                                            and toks + est_tokens(lines[end]) <= MAX_CHUNK_TOKENS)):
            size += len(lines[end]) + 1
            toks += est_tokens(lines[end])
            end += 1
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunks.append(Chunk(rel, f"{label} {start + 1}-{end}", body))
        if end >= n:
            break
        start = max(end - OVERLAP_LINES, start + 1)
    return chunks


# --------------------------------------------------------------------------- #
def _read_csv(path: Path, rel: str, delim: str) -> ReadResult:
    rows = [r for r in csv.reader(io.StringIO(_decode(_read_bytes(path))), delimiter=delim)
            if any(c.strip() for c in r)]
    return ReadResult(rel, "ok" if rows else "empty", _table_chunks(rel, "", rows))


def _table_chunks(rel: str, sheet: str, rows: list) -> list:
    """One overview chunk (header + first rows) plus one key=value chunk per row."""
    if not rows:
        return []
    prefix = f"sheet {sheet}, " if sheet else ""
    header = [h.strip() for h in rows[0]]
    kv_lines = []
    chunks = []
    for i, row in enumerate(rows[1:], start=2):
        pairs = [f"{header[j] if j < len(header) and header[j] else f'col{j + 1}'}={v.strip()}"
                 for j, v in enumerate(row) if v.strip()]
        line = "; ".join(pairs)
        kv_lines.append(line)
        chunks.append(Chunk(rel, f"{prefix}row {i}", line))
    overview = " | ".join(h for h in header if h)
    overview += ("\n" + "\n".join(kv_lines[:CSV_OVERVIEW_ROWS])) if kv_lines else ""
    chunks.insert(0, Chunk(rel, f"{prefix}overview", overview.strip()))
    return chunks


# --------------------------------------------------------------------------- #
_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref or "A")
    n = 0
    for ch in (letters.group(0) if letters else "A"):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


_BUILTIN_FMT = {1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00", 9: "0%", 10: "0.00%"}


def _xlsx_style_formats(zf) -> list:
    """Style index -> number format code, so a cell displayed as 38.50 is not read as 38.5."""
    if "xl/styles.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/styles.xml"))
    custom = {int(n.get("numFmtId")): n.get("formatCode", "") for n in root.iter(_MAIN + "numFmt")}
    xfs = root.find(_MAIN + "cellXfs")
    out = []
    for xf in (xfs if xfs is not None else []):
        fid = int(xf.get("numFmtId", "0"))
        out.append(custom.get(fid) or _BUILTIN_FMT.get(fid) or ("date" if 14 <= fid <= 22 or 45 <= fid <= 47 else ""))
    return out


def _format_number(raw: str, code: str) -> str:
    if not code or code == "General" or "E+" in code.upper():
        return raw
    try:
        x = float(raw)
    except ValueError:
        return raw
    bare = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", code)
    if code == "date" or re.search(r"[ymd]", bare, re.I):
        from datetime import date, timedelta
        return (date(1899, 12, 30) + timedelta(days=int(x))).isoformat()
    pct = "%" in bare
    m = re.search(r"0\.(0+)", bare)
    dec = len(m.group(1)) if m else 0
    v = x * 100 if pct else x
    text = f"{v:,.{dec}f}" if "," in bare else f"{v:.{dec}f}"
    return text + ("%" if pct else "")


def _read_xlsx(path: Path, rel: str) -> ReadResult:
    chunks = []
    with zipfile.ZipFile(io.BytesIO(_read_bytes(path))) as zf:
        names = set(zf.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(_MAIN + "t"))
                      for si in root.iter(_MAIN + "si")]
        style_fmts = _xlsx_style_formats(zf)
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        for sh in wb.iter(_MAIN + "sheet"):
            try:
                target = (targets.get(sh.get(_REL + "id")) or "").lstrip("/")
                if not target:
                    continue
                part = target if target.startswith("xl/") else "xl/" + target
                root = ET.fromstring(zf.read(part))
                rows = []
                for row in root.iter(_MAIN + "row"):
                    cells = {}
                    for c in row.iter(_MAIN + "c"):
                        t, v = c.get("t"), c.find(_MAIN + "v")
                        if t == "inlineStr":
                            val = "".join(x.text or "" for x in c.iter(_MAIN + "t"))
                        elif v is None or v.text is None:
                            continue
                        elif t == "s":
                            val = shared[int(v.text)]
                        elif t == "b":
                            val = "TRUE" if v.text == "1" else "FALSE"
                        else:
                            val = v.text
                            si = c.get("s")
                            if t in (None, "n") and si is not None and int(si) < len(style_fmts):
                                val = _format_number(val, style_fmts[int(si)])
                        cells[_col_index(c.get("r"))] = val
                    if cells:
                        rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
                chunks.extend(_table_chunks(rel, sh.get("name") or "sheet", rows))
            except Exception:
                continue                      # one broken sheet must not hide the others
    return ReadResult(rel, "ok" if chunks else "empty", chunks)


# --------------------------------------------------------------------------- #
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _para_text(el) -> str:
    out = []
    for node in el.iter():
        if node.tag == _W + "t":
            out.append(node.text or "")
        elif node.tag == _W + "tab":
            out.append("\t")
        elif node.tag in (_W + "br", _W + "cr"):
            out.append("\n")
    return "".join(out).strip()


def _docx_entries(body) -> list:
    """Return (kind, text) for each paragraph / table row, in document order."""
    entries = []
    for el in body:
        if el.tag == _W + "sdt":
            content = el.find(_W + "sdtContent")
            if content is not None:
                entries.extend(_docx_entries(content))
        elif el.tag == _W + "p":
            text = _para_text(el)
            if not text:
                continue
            style = el.find(f"{_W}pPr/{_W}pStyle")
            val = (style.get(_W + "val") or "").lower() if style is not None else ""
            entries.append(("heading" if val.startswith(("heading", "title")) else "para", text))
        elif el.tag == _W + "tbl":
            for tr in el.iter(_W + "tr"):
                cells = [" ".join(_para_text(p) for p in tc.iter(_W + "p")).strip()
                         for tc in tr.findall(_W + "tc")]
                if any(cells):
                    entries.append(("row", " | ".join(cells)))
    return entries


def _read_docx(path: Path, rel: str) -> ReadResult:
    chunks = []
    with zipfile.ZipFile(io.BytesIO(_read_bytes(path))) as zf:
        body = ET.fromstring(zf.read("word/document.xml")).find(_W + "body")
        entries = _docx_entries(body) if body is not None else []
        heading, buf, size = "", [], 0

        def flush():
            nonlocal buf, size
            if buf:
                text = (f"[{heading}]\n" if heading else "") + "\n".join(buf)
                chunks.append(Chunk(rel, f"section: {heading}" if heading else f"part {len(chunks) + 1}", text))
            buf, size = [], 0

        for kind, text in entries:
            if kind == "heading":
                flush()
                heading = text
                continue
            if size + len(text) > CHUNK_CHARS and buf:
                flush()
            buf.append(text)
            size += len(text) + 1
        flush()
        extra = []                            # headers, footers, footnotes, comments
        for name in sorted(zf.namelist()):
            if re.match(r"word/(header|footer|footnotes|endnotes|comments)\d*\.xml$", name):
                try:
                    t = _para_text(ET.fromstring(zf.read(name)))
                    if t:
                        extra.append(t)
                except Exception:
                    pass
        if extra:
            chunks.append(Chunk(rel, "headers/footers/notes", "\n".join(extra)))
    return ReadResult(rel, "ok" if chunks else "empty", chunks)


def _read_pptx(path: Path, rel: str) -> ReadResult:
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    chunks = []
    with zipfile.ZipFile(io.BytesIO(_read_bytes(path))) as zf:
        slides = sorted((n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[0]))
        for n in slides:
            root = ET.fromstring(zf.read(n))
            text = "\n".join(t.text for t in root.iter(a + "t") if t.text)
            if text.strip():
                num = re.findall(r"\d+", n)[0]
                chunks.append(Chunk(rel, f"slide {num}", text))
    return ReadResult(rel, "ok" if chunks else "empty", chunks)


# --------------------------------------------------------------------------- #
def _read_pdf(path: Path, rel: str) -> ReadResult:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(_read_bytes(path)))
    if reader.is_encrypted:
        # A PDF that needs a password we do not have is unreadable. Do NOT try to
        # crack it: the correct answer to a question that depends on it is a refusal.
        return ReadResult(rel, "encrypted", [], "password required, skipped")
    chunks, pages = [], 0
    for i, page in enumerate(reader.pages, start=1):
        pages += 1
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            chunks.extend(_chunk_lines(rel, text, f"page {i}, lines"))
    if not chunks:
        return ReadResult(rel, "scanned" if pages else "empty", [],
                          "no text layer (scanned?), needs OCR or a vision model")
    return ReadResult(rel, "ok", chunks)


# --------------------------------------------------------------------------- #
def _read_image(path: Path, rel: str, image_reader) -> ReadResult:
    from PIL import Image
    with Image.open(path) as im:
        im.verify()                           # corrupt images raise -> "unreadable"
    text = ""
    if image_reader is not None:
        try:
            text = (image_reader(path) or "").strip()
        except Exception:
            text = ""
    chunk = Chunk(rel, "image", text or "[image file, contents not read as text]", kind="image")
    note = "text read from image" if text else "no text read; a vision model must look at it"
    return ReadResult(rel, "image", [chunk], note)
