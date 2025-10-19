# 🎥 AI Video Content Analyzer

# AI Video Analyzer 설치 및 실행 매뉴얼

이 문서는 Docker 환경에서 **AI Video Analyzer**를 설치하고 실행하는 방법을 안내합니다.

---

## 1. 요구사항

- Docker >= 24
- docker-compose >= 2.10
- GPU 사용 권장 (CUDA 지원), CPU도 가능하지만 속도 저하
- 최소 24GB 이상 VRAM 권장

---

## 2. 프로젝트 구성

```bash
video-analyzer/
├─ .gitignore
├─ run-gradio.py # 그라디오 UI 실행
│─ qwen-vl-utils/ # QwenVL2의 주요 기능 집합
│─ web_demo_streaming/ # 예제 폴더로 거의 미사용
├─ requirements.txt # 프로그램 관련 설치 될 파이썬 라이브러리들 정보
├─ docker-compose.yml # 도커 컴포즈 설정 파일
├─ Dockerfile # 도커 설정 파일
├─ docker-build.cmd # 1) 빌드해서 실행: 로컬에서 컨테이너 이미지 빌드
├─ docker-up.cmd # 컨테이너 이미지 실행
├─ docker-start.cmd # 중단된 컨테이너 다시 시작
├─ docker-stop.cmd # 컨테이너 서비스 일시 중지
├─ docker-down.cmd # 서비스 컨테이너를 완전히 제거
├─ docker-pull.cmd # 2) 빌드안하고 Pull 받아서 실행: kmong 개발자 개인 docker hub 저장소에서 받아서 하시고 싶은 경우
├─ docker-push.cmd # 3) 현재 버전의 도커 이미지를 개인 저장소에 push해서 저장하고 싶은 경우, 파일 내 경로 수정하여 사용
├─ tail-logs.cmd # 컨테이너 이미지내 로그 확인
├─ models/ # Vision Language 모델 캐시/다운로드 폴더
├─ videos/ # 업로드 된 비디오 저장 폴더
├─ outputs/ # 영상 분석 결과 텍스트 파일 저장 폴더
└─ README.md # 애플리케이션 설치 안내 파일

> `models`, `videos`, `outputs` 폴더는 호스트와 컨테이너 간 볼륨으로 자동 마운트됩니다.

```

## 3. Docker 이미지 빌드

```cmd
docker-build.cmd
```
- Dockerfile 기반으로 PyTorch, CUDA, 필수 라이브러리, 프로젝트 코드를 포함한 이미지 생성
- 모델은 start-model.py 실행 시 캐시된 경로(./models)에 다운로드됨

## 4. 컨테이너 실행

```cmd
docker-run.cmd
```
기본 포트: 7860

- Gradio UI 접속: http://localhost:7860
- 외부 접속 시 서버 IP 사용 가능
- 필요 시 docker-compose.yml에서 GRADIO_SHARE=1 환경변수로 외부 공유 가능

## 5. 컨테이너 중지
```
docker-stop.cmd
```

## 6. 컨테이너 로그 확인
```
tail-logs.cmd
```

## 7. 볼륨 및 환경 변수
| 호스트 폴더   | 컨테이너 경로                         | 설명      |
| ----------- |-------------------------------------|-----------|
| `./models`  | `/workspace/video_analyzer/models`  | Vision Language 모델 다운로드/저장 |
| `./videos`  | `/workspace/video_analyzer/videos`  | 업로드 된 영상 |
| `./outputs` | `/workspace/video_analyzer/outputs` | 분석된 텍스트 |

### 환경 변수
- CUDA_VISIBLE_DEVICES=0 : GPU 선택
- HF_HOME, TRANSFORMERS_CACHE, HF_HUB_CACHE : 모델 캐시 경로
- GRADIO_SHARE=1 : Gradio 외부 공유 옵션

## 8. 사용 방법
#### 1. Gradio 웹 UI 접속
#### 2. 비디오 업로드
#### 3. System Prompt, Max Tokens, FPS 등 설정
#### 4. 🚀 영상 분석 클릭
#### 5. 분석 완료 후 outputs 폴더에 .txt 파일 자동 저장
#### - 분석 텍스트 파일명은 업로드한 영상 파일명과 동일합니다.

## 9. 주의 사항
- 긴 영상(1분 이상) 사용 시 VRAM 부족 가능
- FPS, Max Tokens 조절로 GPU 메모리 사용 최적화 가능
- Docker 환경에서는 /workspace/video_analyzer 경로 기준으로 볼륨이 마운트되어야 정상 동작

## 10. 📌 서비스 제공 전문가 연락처
```
이메일 : ex.friend.ai@gmail.com 또는 methodtweak@naver.com
전화번호 : 010-8865-7020
```

