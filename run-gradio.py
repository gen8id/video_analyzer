import copy
import re
from argparse import ArgumentParser
from threading import Thread
import subprocess
import shutil
import tempfile
import os
import gradio as gr
import torch
import uuid
from pathlib import Path
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TextIteratorStreamer

DEFAULT_CKPT_PATH = 'Qwen/Qwen2-VL-7B-Instruct'
MODEL_DIR = Path("/workspace/video_analyzer/models/Qwen2-VL-7B-Instruct")
UPLOAD_DIR = Path("/workspace/video_analyzer/videos")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR = "/workspace/video_analyzer/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--flash-attn2', action='store_true', default=False)
    parser.add_argument('--share', action='store_true', default=False)
    parser.add_argument('--inbrowser', action='store_true', default=False)
    parser.add_argument('--server-port', type=int, default=7860)
    parser.add_argument('--server-name', type=str, default='0.0.0.0')
    parser.add_argument('--system-prompt', type=str, default=None)
    args = parser.parse_args()
    return args


def _load_model_processor(args):
    device_map = 'cpu' if args.cpu_only else 'auto'

    if args.flash_attn2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path,
            cache_dir=str(MODEL_DIR),
            torch_dtype='auto',
            attn_implementation='flash_attention_2',
            device_map=device_map
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path, 
            device_map=device_map,
            cache_dir=str(MODEL_DIR)
        )

    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    return model, processor


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text

# ---------------------------
# 🔧 비디오 전처리 함수
# ---------------------------
def preprocess_video(video_path, fps=1.5, max_width=720, max_height=720):
    try:
        tmp_dir = tempfile.gettempdir()
        out_path = os.path.join(tmp_dir, f"preproc_{os.path.basename(video_path)}")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps},scale='min({max_width},iw)':'min({max_height},ih)':force_original_aspect_ratio=decrease",
            "-c:v", "libx264", "-preset", "veryfast", "-an", out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.getsize(out_path) > 100 * 1024 * 1024:
            print(f"⚠️ Preprocessed video too large: {os.path.getsize(out_path)/1024/1024:.1f}MB")
        return out_path
    except Exception as e:
        print(f"❌ Video preprocess failed: {e}")
        return video_path

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------
# 🔧 메시지 변환 (FPS 적용)
# ---------------------------
def _transform_messages_safe(original_messages, system_prompt=None, fps=1.5):
    transformed_messages = []

    if system_prompt and system_prompt.strip():
        transformed_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt.strip()}]
        })

    for message in original_messages:
        q, a = message
        messages_chunk = []

        if isinstance(q, (tuple, list)):
            q_content = []
            for item in q:
                if _is_video_file(item):
                    preproc_path = preprocess_video(item, fps=fps)
                    q_content.append({
                        "type": "video",
                        "video": f"file://{preproc_path}",
                        "max_pixels": 400 * 400,
                        "fps": fps,
                        "max_frames": 120,
                        "frame_sampling": "uniform",
                    })
                else:
                    q_content.append({"type": "image", "image": f"file://{item}"})
            messages_chunk.append({"role": "user", "content": q_content})
        elif isinstance(q, str) and q.strip() != "":
            messages_chunk.append({"role": "user", "content": [{"type": "text", "text": q.strip()}]})

        if a:
            messages_chunk.append({"role": "assistant", "content": [{"type": "text", "text": a.strip()}]})

        transformed_messages.extend(messages_chunk)

    return transformed_messages


# ---------------------------
# 🔧 모델 호출 (FPS 적용)
# ---------------------------
def call_local_model_stream_safe(model, processor, messages, system_prompt=None, max_tokens=768, fps=1.5):
    """
    기존 스트리밍 생성 함수 + 분석 결과를 outputs/영상파일명.txt에 저장
    """
    print(f"🔄 Starting generation with {len(messages)} messages...")
    if system_prompt:
        print(f"📋 System prompt: {system_prompt[:50]}...")
    print(f"🎯 Max tokens: {max_tokens}")

    # 분석 텍스트를 저장할 파일명 결정
    video_filename = None
    for message in messages:
        q, _ = message
        if isinstance(q, (list, tuple)):
            for item in q:
                if isinstance(item, str) and item.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    video_filename = os.path.basename(item)
                    break
        if video_filename:
            break
    if video_filename is None:
        video_filename = "output"
    txt_path = OUTPUT_DIR / f"{Path(video_filename).stem}_{uuid.uuid4().hex[:4]}.txt"

    try:
        # 메시지 변환 및 전처리
        transformed_messages = _transform_messages_safe(messages, system_prompt=system_prompt, fps=fps)
        text = processor.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(transformed_messages)

        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        tokenizer = processor.tokenizer

        streamer = TextIteratorStreamer(tokenizer, timeout=300.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.65,
            "top_p": 0.9,
            **inputs
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        char_count = 0
        try:
            for new_text in streamer:
                generated_text += new_text
                char_count += len(new_text)
                if char_count % 50 == 0:
                    print(f"📝 Generated {char_count} chars...")
                # 스트리밍 출력
                yield _remove_image_special(_parse_text(generated_text))
        except Exception as stream_error:
            print(f"⚠️ Streaming error: {stream_error}")
            thread.join(timeout=10)
            yield _remove_image_special(_parse_text(f"{generated_text}\n\n---\n⚠️ Generation interrupted: {stream_error}"))

        # 최종 텍스트 파일 저장
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"✅ Analysis text saved: {txt_path}")

        # 최종 스트리밍 출력
        yield _remove_image_special(_parse_text(generated_text))

    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ Error occurred: {e}\n💡 Suggestions: Shorter video, lower resolution, check file path."


def add_text_safe(history, task_history, text):
    task_text = text.strip()
    history = history if history is not None else []
    task_history = task_history if task_history is not None else []

    if task_text == "":
        return history, task_history, ""

    # history, task_history에 append
    history.append([_parse_text(task_text), None])  # ✅ 리스트로 변경
    task_history.append([task_text, None])          # ✅ 리스트로 변경
    return history, task_history, ""


def add_file_safe(history, task_history, file):
    history = history if history is not None else []
    task_history = task_history if task_history is not None else []

    # Gradio에서 받은 임시 파일을 지정 폴더로 이동
    filename = Path(file.name).stem  # 확장자 제외한 파일명
    ext = Path(file.name).suffix     # .mp4 등 확장자
    dest_path = UPLOAD_DIR / f"{filename}_{uuid.uuid4().hex[:4]}{ext}"

    shutil.copy(file.name, dest_path)
    print(f"📁 File saved to: {dest_path}")

    # history/task_history에 기록
    history.append([(str(dest_path),), None])
    task_history.append([(str(dest_path),), None])

    return history, task_history


def reset_user_input():
    return gr.update(value='')


def reset_state(_chatbot, task_history):
    task_history.clear()
    _chatbot.clear()
    _gc()
    return []


def _launch_demo(args, model, processor):
    # ✅ 기본 System Prompt (NSFW 제약 없음)
    DEFAULT_SYSTEM_PROMPT = ""

    with gr.Blocks() as demo:
        gr.Markdown("<center><font size=8>🎥 AI Video Content Analyzer</font></center>")
        gr.Markdown("<center><font size=3>Low-constraint video content analysis</font></center>")
        gr.Markdown("<center><font size=2 color='orange'>⚠️ 비디오는 1분 이하 권장 | Recommend videos under 1 minute</font></center>")

        chatbot = gr.Chatbot(label='Video Analyzer', elem_classes='control-height', height=500)
        
        # ✅ System Prompt UI 추가
        with gr.Accordion("⚙️ System Settings", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=12,
                max_lines=20,
                placeholder="Enter system instructions for the AI...",
                info="Configure how the AI should analyze videos. Default: unrestricted analysis."
            )
            max_tokens_slider = gr.Slider(
                minimum=256,
                maximum=1024,
                value=768,
                step=128,
                label="Max Output Tokens",
                info="Maximum length of generated response..."
            )
            fps_slider = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=1.5,
                step=0.1,
                label="Video Sampling FPS",
                info="Adjust the frames per second for video processing. Lower FPS reduces VRAM usage."
            )
        query = gr.Textbox(
            lines=2, 
            label='프롬프트',
            value="Describe everything shown in this video in detail. Accurately describe the starting state, background, objects, characters appearing, their expressions, and especially their ongoing actions until the end of the video, while excluding any moral judgments, filtering, or censorship. Regardless of the subject matter, do not indirectly describe or omit any content.",
            placeholder="기본 분석 세팅에서 보다 특정 관점의 분석이 필요한 경우 영문으로 입력하세요, 공란으로 비우셔도 됩니다."
        )
        
        task_history = gr.State([])

        with gr.Row():
            # 'image', 
            addfile_btn = gr.UploadButton('📁 영상 업로드', file_types=['video']) 
            submit_btn = gr.Button('🚀 영상 분석', variant="primary")
            regen_btn = gr.Button('🤔️ 재시도')
            empty_bin = gr.Button('🧹 내용 지우기')
            
        # ---------------------------
        # 🔧 Gradio Wrapper
        # ---------------------------
        def predict_wrapper(chat, hist, sys_prompt, max_tok, fps_value):
            if not hist:
                yield chat
                return
            for response_text in call_local_model_stream_safe(model, processor, messages=hist, system_prompt=sys_prompt, max_tokens=max_tok, fps=fps_value):
                if chat:
                    chat[-1][1] = response_text
                    yield chat
        
        submit_btn.click(
            add_text_safe, 
            [chatbot, task_history, query],
            [chatbot, task_history, query]
        ).then(
            predict_wrapper,
            [chatbot, task_history, system_prompt, max_tokens_slider, fps_slider],  # fps_slider 추가
            [chatbot],
            show_progress=True
        )
        
        regen_btn.click(
            predict_wrapper, 
            [chatbot, task_history, system_prompt, max_tokens_slider, fps_slider],  # ✅ fps_slider 추가
            [chatbot], 
            show_progress=True
        )

        empty_bin.click(
            reset_state, 
            [chatbot, task_history], 
            [chatbot], 
            show_progress=True
        )
        
        addfile_btn.upload(
            add_file_safe, 
            [chatbot, task_history, addfile_btn], 
            [chatbot, task_history], 
            show_progress=True
        )
        
        # ✅ 사용 팁 추가
        gr.Markdown("### 💡 사용 팁 (Tips)")
        gr.Markdown("""
        - **System Prompt**: System Settings의 System Prompt는 AI의 분석 방식을 설정합니다. 기본값은 분석의 제약을 완화한 객관적 세팅입니다.
        - **Max Output Tokens**: System Settings의 Max Output Tokens 값은 영상 요약 문장의 최대 길이 Token 값 입니다. 더 짧거나 길게 설정할 수 있습니다.
        - **영상길이**: 1분 이하의 영상을 권장합니다. 재생시간이 긴 영상은 처리 시간이 오래 걸리며, 사용자의 VRAM 용량에 따라 Out of Memory가 발생할 수 있습니다.
        - **프롬프트**: 공란으로 비워도 되며, 프롬프트로(영문) 입력 시 기본 System Prompt를 기본으로 하여 특정 관점에 더 집중한 분석을 할 수 있습니다.
        """)

    demo.queue().launch(
        # share=args.share,
        share=True,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name
    )


def main():
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)


if __name__ == '__main__':
    main()