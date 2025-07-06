# R_server.py (최종 완성본)

import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict
import uvicorn
import json
from openai import OpenAI
# BackgroundTasks를 fastapi에서 가져옵니다.
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from inference_engine import ColPaliRetriever 

class LLMOrchestrator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        print("LLMOrchestrator: Initialized with new integrated prompt.")

    def parse_user_request(self, text: str) -> Dict[str, str]:
        print(f"LLM: Parsing text with GPT -> '{text}'")
        
        # 동료분의 아이디어와 기존 후처리 요구사항을 통합한 새로운 프롬프트
        system_instruction = f"""
        너는 주어진 문장에서 '현재 지칭하는 옷(source)'과 '적용하고 싶은 옷(target)'을 찾아서, 후속 작업에 필요한 두 가지 키를 가진 JSON 객체로 반환하는 AI야.
        반환할 JSON의 키는 "pbr_prompt"와 "db_query" 여야 해.

        사용자 문장: "{text}"

        작업 지시:
        1. 문장에서 'source'와 'target'을 정확히 파악해줘.
        2. "pbr_prompt": 파악한 'source'를 영어로 번역하고, "english_word ." 형식으로 만들어줘. (예: '파란색 셔츠' -> 'blue shirt .')
        3. "db_query": 파악한 'target'의 핵심 키워드만 단순 문자열로 만들어줘. (예: '청바지 드레스' -> '청바지 드레스')
        4. 만약 source나 target을 찾을 수 없다면 해당 값은 null로 설정해줘.
        5. 다른 설명 없이 최종 JSON 객체만 반환해.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_instruction}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            pbr_prompt = result.get("pbr_prompt")
            db_query = result.get("db_query")

            # Null 값에 대한 기본값 처리
            if not pbr_prompt: pbr_prompt = "object ."
            if not db_query: db_query = ""
            
            print(f"LLM -> PBR Prompt: '{pbr_prompt}', DB Query: '{db_query}'")
            return {"pbr_prompt": pbr_prompt, "db_query": db_query}

        except Exception as e:
            print(f"LLM ERROR: Failed to parse with GPT. Error: {e}")
            raise HTTPException(status_code=500, detail="LLM processing failed.")


# --- 초기 설정 및 모델 로딩 ---
print("🚀 서버 초기화를 시작합니다...")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OUTPUT_DIR = Path("MaterialPalette") / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_FOLDER_PATH = "dress-data" 
retriever = ColPaliRetriever(db_folder_path=DB_FOLDER_PATH, model_name="tsystems/colqwen2.5-3b-multilingual-v1.0")
llm_orchestrator = LLMOrchestrator(api_key=OPENAI_API_KEY)

app = FastAPI()
print("✅ 서버 초기화 완료. 요청을 기다립니다.")

@app.get("/")
async def get_test_page():
    return FileResponse("test.html")

@app.post("/apply-style")
async def apply_style(
    background_tasks: BackgroundTasks, 
    text: str = Form(...), 
    image: UploadFile = File(...)
):
    print(f"\n\n--- 새로운 요청 수신 ---")
    print(f"입력 텍스트: '{text}'")
    print(f"입력 이미지: '{image.filename}'")

    # --- 파일 이름 규칙 변경 ---
    image_basename = Path(image.filename).stem 
    
    # 최종 PBR 작업 폴더 경로: MaterialPalette/output/<파일명>/
    pbr_working_dir = OUTPUT_DIR / image_basename
    os.makedirs(pbr_working_dir, exist_ok=True)

    target_image_path = pbr_working_dir / image.filename
    with open(target_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    print(f"PBR 파이프라인을 위해 이미지를 저장했습니다: {target_image_path}")

    try:
        # --- 빠른 작업: LLM 분석 및 ColPali 검색 ---
        parsed_keywords = llm_orchestrator.parse_user_request(text)
        pbr_prompt = parsed_keywords["pbr_prompt"]
        db_query = parsed_keywords["db_query"]

        print(f"\n[Immediate Task] DB에서 '{db_query}' 키워드로 검색합니다...")
        retrieved_items = retriever.search(query_text=db_query, k=1)
        if not retrieved_items:
            raise HTTPException(status_code=404, detail=f"'{db_query}'에 해당하는 아이템을 DB에서 찾을 수 없습니다.")
        
        target_filename = retrieved_items[0]['metadata']['image_name']
        print(f"✅ [Immediate Task] DB 검색 완료! Target Filename: {target_filename}")

        # --- 느린 작업: PBR 생성을 백그라운드로 전달 ---
        print("\n[Background Task] PBR 생성을 백그라운드 작업으로 예약합니다...")
        background_tasks.add_task(run_pbr_generation_pipeline, target_image_path, pbr_prompt)

        # --- 즉시 응답 ---
        final_response = {"target_filename": target_filename}
        print("\n요청 처리 성공. Unity로 즉시 응답을 전송합니다. PBR 생성은 백그라운드에서 계속됩니다.")
        return JSONResponse(content=final_response)

    except Exception as e:
        # 주요 로직에서 발생한 오류는 여기서 처리
        print(f"🚨 즉시 처리 중 오류 발생: {e}")
        # 오류 발생 시 생성된 폴더 정리
        if 'pbr_working_dir' in locals() and pbr_working_dir.exists():
            shutil.rmtree(pbr_working_dir)
            print(f"오류 발생으로 임시 PBR 작업 폴더를 삭제했습니다: {pbr_working_dir}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

def run_pbr_generation_pipeline(image_path: Path, prompt: str):
    """
    PBR 생성 파이프라인 전체를 실행하는 함수. 이제 백그라운드에서 실행됩니다.
    """
    print(f"\n--- InBackground: PBR 파이프라인 시작 (이미지: {image_path.name}) ---")
    try:
        command = [
            "python", "run_full_pipeline.py",
            "--image_path", str(image_path),
            "--prompt", prompt
        ]
        print(f"InBackground: PBR 파이프라인 실행: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        print("InBackground: PBR 파이프라인 stdout:", process.stdout)
        
        # PBR 결과물은 삭제하지 않고 유지
        print(f"--- ✅ InBackground: PBR 파이프라인 성공적으로 완료 (이미지: {image_path.name}) ---")
        
    except Exception as e:
        # 백그라운드 작업 중 오류 기록
        print(f"--- 🚨 InBackground: PBR 파이프라인 오류 발생 (이미지: {image_path.name}) ---")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"  - Stderr: {e.stderr}")
        else:
            print(f"  - Exception: {e}")
        # 오류가 발생했더라도 생성된 폴더는 디버깅을 위해 일단 남겨둘 수 있습니다.
        # 필요 시 여기에 폴더 삭제 로직 추가 가능: shutil.rmtree(image_path.parent)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)