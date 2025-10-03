# DesignerClip Overview

DesignerClip은 의류 이미지를 입력 받아 세그멘테이션, 멀티모달 검색, MaterialPalette 기반 PBR 맵 생성을 한 번에 수행하는 서버 애플리케이션입니다. **실행 진입점은 `R_server.py`**이며, ngrok 터널을 통해 외부에서 접근할 수 있는 `/apply-style` API를 제공합니다.

## Pipeline Components
- `R_server.py`: FastAPI + uvicorn 서버. 요청 수신, 프롬프트 파싱, 검색, 파이프라인 실행을 순차적으로 조율합니다.
- `LLMOrchestrator`: GPT-4o에 텍스트를 전달해 `pbr_prompt`(MaterialPalette용)와 `db_query`(검색용)를 JSON으로 받습니다. `OPENAI_API_KEY`가 반드시 필요합니다.
- `ColPaliRetriever` (`inference_engine.py`): `dress-data/` 폴더 이미지를 ColQwen2.5 임베딩으로 인덱싱하고, 타깃 이미지를 찾아줍니다.
- `run_full_pipeline.py`: MaterialPalette 서브모듈을 호출해 알베도/노멀/러프니스 맵을 생성합니다.
- `static_maps/`: 세션별로 생성된 PBR 맵을 복사해 두고, ngrok 주소와 조합해 다운로드 URL을 응답합니다.

## Setup Checklist (순서대로)
1. `.env` 또는 터미널에서 `OPENAI_API_KEY`를 설정합니다.
2. 가상 환경 구성 후 의존성을 설치합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. `Grounded-SAM-2/checkpoints/`에 SAM2·Grounded DINO 가중치를, `MaterialPalette/`에 필요한 모델 파일을 배치합니다.
4. 비교 대상 의류 이미지를 `dress-data/`에 저장합니다.
5. ngrok을 설치하고 계정과 연결합니다.
6. ngrok가 발급하는 주소를 반영할 수 있도록 `R_server.py`의 `base_url` 문자열을 수정하거나, 환경 변수로 치환하는 편이 관리에 용이합니다.

## Run Sequence
1. **ngrok 실행**
   ```bash
   ngrok http 8000
   ```
   출력되는 `https://<random>.ngrok.app` 주소를 기억합니다.
2. **서버 코드 업데이트**: `R_server.py`의 `base_url` 값을 위 ngrok URL로 교체합니다.
3. **서버 시작**
   ```bash
   python R_server.py
   ```
   콘솔 로그로 초기화 과정이 출력되고 `http://0.0.0.0:8000`에서 대기합니다. `/` 엔드포인트는 로컬 확인용 `test.html`을 제공합니다.
4. **API 호출**: `POST /apply-style`에 `multipart/form-data`로 `text`(스타일 지시)와 `image`(원본 의류 이미지)를 전송합니다.

## Request Flow (서버 내부 단계)
1. 업로드 이미지를 `MaterialPalette/output/<session_id>/`에 저장합니다.
2. LLM이 텍스트에서 `pbr_prompt`와 `db_query`를 추출합니다.
3. `ColPaliRetriever`가 `dress-data/`에서 가장 유사한 타깃 이미지를 선택합니다.
4. `run_full_pipeline.py`가 MaterialPalette를 실행해 세 가지 PBR 맵을 생성합니다.
5. 생성된 파일을 `static_maps/<session_id>/`로 복사하고, ngrok `base_url`과 결합해 URL을 만듭니다.
6. 응답 JSON에 `target_filename`과 세 맵 URL을 담아 반환한 뒤, 임시 작업 폴더를 정리합니다.

## Troubleshooting
- MaterialPalette 실행 실패 시 `subprocess.run(..., capture_output=True)`를 통해 stderr가 콘솔에 출력됩니다. 먼저 해당 오류 메시지를 확인하세요.
- GPU가 없거나 메모리가 부족하면 모델 로딩 단계에서 실패할 수 있습니다. 필요 시 `device="cpu"` 처리를 추가하십시오.
- `/maps` 경로는 외부에 노출되므로, 세션 폴더를 주기적으로 정리하거나 접근 제어를 구성하는 것이 좋습니다.

## Additional Notes
- 실험용 Gradio UI(`app.py`, `main.py`)는 여전히 포함되어 있지만, 프로덕션 플로우는 `R_server.py`를 기준으로 문서화되어 있습니다.
- 기여 절차와 스타일 규칙은 루트의 `AGENTS.md`에서 확인하세요.
