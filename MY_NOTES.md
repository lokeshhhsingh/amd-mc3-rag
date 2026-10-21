# MC3 progress notes

## Done (needs no GPU, no Docker)
- `starter/app/rag/readers.py`  reads pdf, docx, xlsx, csv, txt/log/py, pptx, images. Never raises: unreadable,
  encrypted, binary, empty and corrupt files are skipped and reported, the walk keeps going.
- `starter/app/rag/search.py`   BM25 search (stdlib only). Exact ids (TQ-40, ORR-1847, THERM_ALERT#) match strongly.
  Withdrawn/superseded files are down-weighted. A second hop follows ids found in the best chunks (multi-hop questions).
- `starter/app/rag/llm.py`      prompt, answer cleaning, grounding check (refuse if the answer is in no excerpt),
  citations built from the excerpts the model says it needed.
- `starter/app/app.py`          index() and answer() filled in; the harness plumbing is untouched.
- `eval_retrieval.py`, `check_logic.py`  offline checks. Run them as a NON-root user (see below).

## Also written but NOT yet run on a GPU
- `model_server.py` (resident model behind a unix socket, started detached by --index), `Dockerfile` (ships Qwen3-VL-8B weights),
  `score_samples.py`, `RUNSHEET.md` (exact commands). Server lifecycle is tested with a fake model in `check_logic.py` parts 3-4.

## Still to do (needs a GPU)
1. Resident model process on a unix socket (MC3_LLM_SOCKET) so each question does not reload the model.
2. Vision model: caption/transcribe images at index time (`rag/vision.py`) and attach them at question time.
   Tesseract OCR is only a local-testing fallback and will not be in the container.
3. Dockerfile: ship model weights in the image, start the resident process, add `pypdf` (already in requirements.txt).
4. `selfcheck.py` on the built image, test with `--network none` and `--cap-drop DAC_OVERRIDE`.

## Testing note
Run `sh setup.sh` first. As root, chmod 000 does not block reading, so test as another user, for example:
    setpriv --reuid=65534 --regid=65534 --clear-groups python3 eval_retrieval.py mc3-corpus

## Free-GPU model test
`mc3_colab_model_test.ipynb` runs the real code with Qwen3-VL-4B on a free Colab T4 (needs `MC3_DTYPE=float16`).
It validates the prompt, image reading and answer rules, not the AMD image.

## Result of the free-GPU test (Qwen3-VL-4B, Colab T4)
9/10 sample questions fully correct on first run. The miss (Q9, two-file answer) was a citation miss:
the model cited only the CSV, not the log that named the ticket. `patch_multihop.py` fixes it deterministically
(a chunk found by following an id also cites the chunk that named the id) and is already applied in this kit.

## Hard practice set
`build_hard_corpus.py` writes `mc3-corpus-hard/` + `hard-questions.json` (32 questions: look-alike docs, 4000-line log, spreadsheets
with number formats, images with meaningless names, two-hop lookups, unanswerable and unreadable cases).
Offline retrieval coverage: 32/32. Run the model on it with `mc3_colab_test_v2.ipynb`.

## Result of the hard-set run (Qwen3-VL-4B, T4): 22/31, then fixes (kit v3)
- 6 questions hit CUDA out-of-memory on the 15 GB T4 with big prompts -> prompt capped (14 excerpts, 15,000 chars, 1,000 chars each); GPU cache freed per request.
- Model cited the wrong file (right answer) -> citations are now VERIFIED: a cited excerpt must contain the answer or share an identifier with one that does.
- "+/-0.04 mm" refused by the grounding check -> sign dropped from numeric answers, grounding is token-boundary aware.
- Asked about a product that does not exist (HX-15) the model answered with HX-12's value -> prompt rule + a guard that refuses when a hyphenated id in the question appears nowhere in the corpus.

## Hard-set run #2 (T4): 25/31, then kit v4
- Q11/12/27 out-of-memory: Qwen reads each DIGIT as a token, so logs cost ~1 token/char. Budgets are now in TOKENS (default 2800, 6000 in the
  Dockerfile for the MI300X), chunks are token-capped at index time, max 2 chunks per file in the prompt, and an OOM triggers a retry with half the excerpts.
- Q10/Q26 cited an unrelated file that also held the value -> when the question names an id (HRT-1187, HX-ERR-4471) the cited excerpt must contain it.
- Q2 was a scorer bug (units): the local scorer now normalises like the grader.
- Two-hop retrieval: seeds are the best chunk of each of the best FILES; per-level hop budgets; hops ranked by question relevance.
- A prompt line I added ("must be about EXACTLY the product") made the 4B model over-refuse (sample Q2): removed. The code-side guard stays.

## Hard-set run #3 (T4): sample 10/10, hard 28/31, then kit v5
- Misses: Q9 (model answered about E102's neighbour), Q11 (right answer, missed the log citation), Q12 (3-hop; chose the 2025 price).
- `oracle_check.py` plays a perfect model to test the CITATION logic alone; it found real bugs: spurious value matches (03:15:09 is not 15),
  bridging through common ids (TQ-40), per-file caps blocking the answer row, wrong hop ranking.
- Citations now come from a LINK WALK: a link is an excerpt with no answer of its own that shares exactly ONE rare id (not in the question or answer)
  that sits right next to the answer or the previous link; chains are kept only if they end at a question-matched excerpt.
- Excerpts are chosen by relevance per token (top two direct hits always in). Entity check: if the answer's files never mention the code the question
  asks about (E101), ask once more showing only excerpts that do. Prompt rule: use the newest year/revision unless the question names one.

## Hard-set run #4 (T4, kit v5): sample 10/10, hard 29/31 -> kit v6
- Q5: answered from the DRAFT datasheet -> drafts (DRAFT in caps, 'draft' in the file name, 'not approved') are now flagged like withdrawn files:
  ranking penalty + a prompt warning. An approved brief that merely MENTIONS a draft is not flagged.
- Q10: cited a ticket row that never names the asked-about code -> when the question names an id, the holder that contains it replaces the cited excerpt.
