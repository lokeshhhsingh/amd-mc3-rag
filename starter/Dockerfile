# Your image MUST be built on the ROCm base image AMD mandates (checked by layer identity).
# It ships torch and PIL only, and supports gfx942 (MI300X) and gfx1100 (Navi 31).
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# 1. Python packages. Pin torch and friends to what the base image already has, so pip can never
#    swap in a different (CUDA) build, then fail the build loudly if torch lost its ROCm support.
COPY app/requirements.txt /app/requirements.txt
RUN pip freeze | grep -i -E '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > /tmp/pins.txt || true; \
    pip install --no-cache-dir -r /app/requirements.txt -c /tmp/pins.txt \
 && python3 -c "import torch; assert torch.version.hip, 'torch lost its ROCm build'; print('torch', torch.__version__, 'hip', torch.version.hip)"

# 2. Weights. Downloaded NOW, at build time while the network is open, then shipped inside the image
#    and used OFFLINE. At evaluation there is no network. Counts against the 60 GiB image limit.
ARG MODEL_REPO=Qwen/Qwen3-VL-8B-Instruct
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}', local_dir='/models/qwen3-vl-8b', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])" \
 && du -sh /models/qwen3-vl-8b
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MC3_MODEL_DIR=/models/qwen3-vl-8b
# Prompt budget in tokens. The default (2800) is sized for a 15 GB card; an MI300X (192 GB) can afford much more.
ENV MC3_MAX_PROMPT_TOKENS=6000

# 3. Your code.
COPY app/ /app/

# The harness runs `python3 /app/app.py --index /app/corpus` once, then one process per question.
# --index starts model_server.py (detached) which keeps the model on the GPU behind a unix socket.
RUN mkdir -p /app/corpus /app/output
CMD ["sleep", "infinity"]
