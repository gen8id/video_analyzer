# Base image — 최신 PyTorch + CUDA + cuDNN 포함 버전
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

WORKDIR /workspace/video_analyzer

RUN apt-get update && apt-get install -y wget xz-utils && \
    cd /tmp && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xJf ffmpeg-release-amd64-static.tar.xz && \
    cd ffmpeg-*-amd64-static && \
    mv ffmpeg ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    cd / && rm -rf /tmp/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install flash-attn==2.6.1 || echo "Flash-attn 설치 실패해도 무시"

# 소스 전체 복사
COPY *.py ./
COPY qwen-vl-utils ./qwen_vl_utils
COPY web_demo_streaming ./web_demo_streaming

# Dockerfile
RUN mkdir -p /workspace/video_analyzer/models /workspace/video_analyzer/videos /workspace/video_analyzer/outputs

ENV HF_HOME=/workspace/video_analyzer/models
ENV HF_HUB_CACHE=/workspace/video_analyzer/models

EXPOSE 7860

ENTRYPOINT ["python", "run-gradio.py", "--share"] 
