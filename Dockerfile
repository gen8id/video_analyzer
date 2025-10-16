# Base image — 최신 PyTorch + CUDA + cuDNN 포함 버전
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

# 작업 디렉토리 설정
WORKDIR /workspace/video_analyzer

# 시스템 패키지 업데이트 및 필수 의존성 설치
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 wget curl vim && \
    rm -rf /var/lib/apt/lists/*

# 캐시 최적화를 위해 requirements.txt만 먼저 복사
COPY requirements.txt .

# pip 최신화 및 캐시 방지 설정
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# flash-attn 주석 포함 설치
RUN pip install flash-attn==2.6.1 || echo "Flash-attn 설치 실패해도 무시"

# 프로젝트 소스 복사
COPY scripts/ ./scripts/

# 모델 다운로드 디렉토리 생성
RUN mkdir -p models videos outputs

# 환경 변수 (모델 캐시 경로 등)
ENV HF_HOME=/workspace/models
ENV TRANSFORMERS_CACHE=/workspace/models
ENV HF_HUB_CACHE=/workspace/models

# Gradio UI 포트
EXPOSE 7860

# Start-up 스크립트 실행
ENTRYPOINT ["python", "scripts/start-model.py"]
