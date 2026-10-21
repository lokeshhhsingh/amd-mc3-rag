# MC3 run sheet: build, test, push, submit

Do this on a machine that has Docker and an AMD GPU (an AMD Developer Cloud droplet is ideal).
Replace YOURUSER with your Docker Hub username.

## 0. First checks on the machine
    docker --version                 # if missing: install Docker (Ubuntu: apt install docker.io), then: sudo usermod -aG docker $USER
    rocm-smi                         # the GPU should be listed
    df -h /                          # want 100+ GB free
    docker login                     # Docker Hub username + access token

## 1. Unpack the kit and recreate the two zip-proof cases
    unzip mc3-starter-kit-with-reader.zip && cd mc3-starter-kit && sh setup.sh

## 2. Offline tests (no GPU needed, safe to run as a normal user)
    python3 check_logic.py mc3-corpus
    python3 eval_retrieval.py mc3-corpus
    python3 eval_retrieval.py mc3-corpus-hard hard-questions.json     # harder practice set

## 3. Build (pulls ~29 GiB base image, downloads ~17 GB of weights; takes a while)
    docker build -t YOURUSER/mc3-rag:v1 starter/
The build fails loudly if torch lost its ROCm support or the weights did not download.

## 4. Test on the GPU the way the harness runs it
    docker run -d --name mc3test --network none --device=/dev/kfd --device=/dev/dri \
        --group-add video --ipc=host --shm-size=16g --cap-drop DAC_OVERRIDE YOURUSER/mc3-rag:v1
    docker cp mc3-corpus/. mc3test:/app/corpus
    time docker exec mc3test python3 /app/app.py --index /app/corpus       # must finish in 10 min
    python3 score_samples.py mc3test                                       # each question must take < 30 s
    docker exec mc3test tail -n 30 /tmp/mc3_model_server.log               # model load messages, errors
    docker exec mc3test rocm-smi --showmeminfo vram                         # VRAM must be 1-48 GiB
    # harder practice set (32 questions): index it, then score it
    docker exec mc3test sh -c 'rm -rf /app/corpus/* /app/index/*'
    docker cp mc3-corpus-hard/. mc3test:/app/corpus
    docker exec mc3test python3 /app/app.py --index /app/corpus
    python3 score_samples.py mc3test --questions hard-questions.json
    docker rm -f mc3test

## 5. The kit's own checker (it starts the container WITHOUT a GPU, so the model server will
##    decline to start and every answer will be a refusal; here it only checks size, no network, timing and file shape)
    python3 selfcheck.py YOURUSER/mc3-rag:v1 mc3-corpus

## 6. Push, then submit the image NAME on lablab
    docker push YOURUSER/mc3-rag:v1          # the repository must be PUBLIC
Submit `YOURUSER/mc3-rag:v1` (check the submission form for the exact format it wants).
Do NOT delete the image from the registry until the challenge is over.

## 7. Clean up the machine
    docker image rm YOURUSER/mc3-rag:v1 && docker builder prune -a
If it is a droplet, destroy it so credits stop being used.

## Known risks (all UNTESTED: nothing here has run on a GPU yet)
- Qwen3-VL may need `torchvision`, which the base image may not have. If `AutoProcessor` raises an ImportError
  in the server log, check `docker run --rm BASE python3 -c "import torchvision"` and install a torchvision
  that matches the image's torch WITHOUT letting pip replace torch.
- `transformers` must be recent enough for Qwen3-VL (>= 4.57). If the class is missing, raise the pin.
- Per-question time: generation here uses plain transformers (no vLLM). It should be a few seconds per question;
  score_samples.py prints the real number.
- If the model does not follow the JSON format, tighten the prompt in rag/llm.py (RULES). Wrong-format replies
  become safe refusals, never crashes.
- Image names in the graded corpus may be uninformative (IMG_0042.png). Images are transcribed by the model at
  index time precisely so they can still be found by their content.

## Prompt budget
`MC3_MAX_PROMPT_TOKENS` (default 2800, sized for a 15 GB card) is set to 6000 in the Dockerfile for the MI300X. If answers on the AMD GPU
miss evidence that was retrieved, raise it (the 192 GB card can afford much more); if you ever see out-of-memory errors, lower it.
