import os
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_DIR = Path("/workspace/models/Qwen2-VL-7B-Instruct")

# 모델 다운로드 확인
if not MODEL_DIR.exists():
    print(f"📥 Downloading {MODEL_NAME}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_DIR)
    )
    AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_DIR)
    )
else:
    print(f"✅ Model already exists at {MODEL_DIR}")

# Gradio 앱 실행
os.system("python scripts/run-gradio.py --server-port 7860")
