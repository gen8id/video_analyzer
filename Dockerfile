# Base image — 최신 PyTorch + CUDA + cuDNN 포함 버전
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

# 시스템 패키지 업데이트 및 필수 의존성 설치
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 wget curl vim && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace/video_analyzer

# 캐시 최적화를 위해 requirements.txt만 먼저 복사
COPY requirements.txt .

# pip 최신화 및 캐시 방지 설정
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# flash-attn 주석처리 → 선택적으로 설치 가능하도록 ARG 처리
# 필요할 때 빌드 시 다음처럼 실행: 
#   docker build --build-arg INSTALL_FLASH_ATTN=true -t videollama3 .
ARG INSTALL_FLASH_ATTN=false
RUN if [ "$INSTALL_FLASH_ATTN" = "true" ]; then \
      pip install flash-attn==2.6.1; \
    fi

# 소스 복사
COPY . .

# 환경 변수 (모델 캐시 경로 등)
ENV HF_HOME=/workspace/models \
    TRANSFORMERS_CACHE=/workspace/models \
    HF_HUB_CACHE=/workspace/models

# Gradio UI 포트
EXPOSE 7860

# 기본 명령 (필요 시 주석 해제)
# CMD ["python", "run-gradio.py", "--server-port", "7860", "--model-path", "DAMO-NLP-SG/VideoLLaMA3-7B"]
