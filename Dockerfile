# Base image — 최신 PyTorch + CUDA + cuDNN 포함 버전
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

WORKDIR /workspace/video_analyzer

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 wget curl vim && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install flash-attn==2.6.1 || echo "Flash-attn 설치 실패해도 무시"

# 소스 전체 복사
COPY *.py ./
COPY qwen-vl-utils ./qwen_vl_utils
COPY web_demo_streaming ./web_demo_streaming

RUN mkdir -p models videos outputs

ENV HF_HOME=/workspace/models
ENV TRANSFORMERS_CACHE=/workspace/models
ENV HF_HUB_CACHE=/workspace/models

EXPOSE 7860

# ENTRYPOINT ["python", "run-gradio.py"]
