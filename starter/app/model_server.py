#!/usr/bin/env python3
"""Resident model process: loads the vision-language model ONCE onto the GPU, then answers
requests on a unix socket, so each per-question process of app.py only has to talk to it.

Protocol, one JSON object per line each way:
  request : {"prompt": str, "images": [path, ...], "max_new_tokens": int}   or {"ping": 1}
  reply   : {"text": str}                                                  or {"text": "", "error": str}

The socket is bound only AFTER the model has loaded, so "the socket answers a ping" means
"the model is ready". Started detached by app.py (see rag/llm.py), never by hand in evaluation.
Set MC3_FAKE_MODEL=1 to run without torch or a GPU (used by check_logic.py to test the plumbing).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

SOCK = os.environ.get("MC3_LLM_SOCKET", "/tmp/mc3_model.sock")
MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/qwen3-vl-8b")
MAX_IMAGE_SIDE = 1280


def log(msg: str) -> None:
    print(f"[model_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_fake():
    time.sleep(float(os.environ.get("MC3_FAKE_LOAD_SECONDS", "0")))

    def generate(prompt, images, max_new_tokens):
        if images and "Transcribe" in prompt:
            return "FAKE TRANSCRIPTION of " + ", ".join(Path(p).name for p in images)
        return json.dumps({"answer": "", "evidence": []})
    return generate


def load_real():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    if not torch.cuda.is_available():            # ROCm builds of torch also report "cuda"
        raise RuntimeError("no GPU visible to torch: refusing to run on the CPU")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    dtype = getattr(torch, os.environ.get("MC3_DTYPE", "bfloat16"))   # float16 on T4-class GPUs
    try:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=dtype)
    except TypeError:                            # older transformers call it torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    model = model.to("cuda").eval()              # place on the GPU explicitly, never fall back
    log(f"model on GPU, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

    def generate(prompt, images, max_new_tokens):
        content = []
        for p in images:
            im = Image.open(p).convert("RGB")
            im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(model.device)
        try:
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        finally:
            del inputs                           # never let one request's tensors pile up on the GPU
            torch.cuda.empty_cache()
    return generate


def serve(generate) -> None:
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o666)
    srv.listen(8)
    log(f"ready on {SOCK}")
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                data = b""
                while not data.endswith(b"\n"):
                    part = conn.recv(1 << 16)
                    if not part:
                        break
                    data += part
                req = json.loads(data.decode())
                if req.get("ping"):
                    reply = {"text": "pong"}
                else:
                    reply = {"text": generate(req["prompt"], req.get("images", []),
                                              int(req.get("max_new_tokens", 160)))}
            except Exception as exc:             # one bad request must never kill the server
                reply = {"text": "", "error": f"{type(exc).__name__}: {exc}"}
            try:
                conn.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                pass


def main() -> int:
    Path(SOCK + ".pid").write_text(str(os.getpid()))
    try:
        generate = load_fake() if os.environ.get("MC3_FAKE_MODEL") else load_real()
    except Exception as exc:
        log(f"could not load the model: {type(exc).__name__}: {exc}")
        return 1
    serve(generate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
