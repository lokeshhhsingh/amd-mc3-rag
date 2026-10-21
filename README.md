# AMD MC3 RAG

Retrieval-augmented question answering for the Lablab x AMD AI Academy Challenge (Mini Challenge 3).

It reads PDF, Word, Excel, CSV, log, code and image files, answers questions with exact file citations,
and uses a Qwen3-VL vision-language model served with PyTorch on AMD ROCm. It is packaged as a Docker image
built on AMD's ROCm base image.

## Layout
- `starter/` : the container (Dockerfile, readers, search, model server)
- `mc3-corpus/`, `mc3-corpus-hard/` : the sample corpus and my harder practice corpus, with questions
- `check_logic.py`, `eval_retrieval.py`, `oracle_check.py`, `score_samples.py` : tests
- `RUNSHEET.md` : steps to build, test, push and submit the image

## Status
Tested on a free GPU with a smaller Qwen3-VL-4B model on the sample and practice question sets.
The final build and test on an AMD Instinct GPU is pending GPU access.
