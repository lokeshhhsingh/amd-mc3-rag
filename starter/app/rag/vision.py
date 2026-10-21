"""Turns images into text at index time. Returns a callable(Path) -> str, or None.

Order of preference:
  1. The resident vision-language model (model_server.py), when the weights or a running server exist.
  2. Tesseract OCR, if the binary and pytesseract are installed (useful for local testing;
     it is weak on diagrams and photographed labels, so do not rely on it for scoring).
If neither is available the image is still indexed by its file name and the vision model
is shown the image itself when it is retrieved.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def _tesseract_reader():
    if not shutil.which("tesseract"):
        return None
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return None

    def read(path: Path) -> str:
        with Image.open(path) as im:
            return pytesseract.image_to_string(im.convert("RGB"))
    return read


_model_failed = False       # after one failed wait, stop waiting for every further image


def _model_reader():
    from . import llm

    def read(path: Path) -> str:
        global _model_failed
        if _model_failed:
            raise RuntimeError("vision model not available")
        backend = llm.get_backend(wait=420)
        if backend is None:
            _model_failed = True
            raise RuntimeError("vision model did not become ready")
        backend.timeout = 60.0
        return backend.complete(llm.TRANSCRIBE_PROMPT, [path], max_new_tokens=400)
    return read


def get_image_reader():
    from . import llm
    if llm.model_available():
        return _model_reader()
    return _tesseract_reader()
