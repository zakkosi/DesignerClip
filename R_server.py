import os
import uuid
import shutil
import subprocess
import base64
from pathlib import Path
from typing import Dict
import uvicorn
import json
from openai import OpenAI
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from inference_engine import ColPaliRetriever
from fastapi.staticfiles import StaticFiles

# LLMOrchestrator 클래스 및 encode_image_to_base64 함수 (변경 없음)
class LLMOrchestrator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        print("LLMOrchestrator: Initialized with new integrated prompt.")

    def parse_user_request(self, text: str) -> Dict[str, str]:
        print(f"LLM: Parsing text with GPT -> '{text}'")
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
            if not pbr_prompt: pbr_prompt = "object ."
            if not db_query: db_query = ""
            print(f"LLM -> PBR Prompt: '{pbr_prompt}', DB Query: '{db_query}'")
            return {"pbr_prompt": pbr_prompt, "db_query": db_query}
        except Exception as e:
            print(f"LLM ERROR: Failed to parse with GPT. Error: {e}")
            raise HTTPException(status_code=500, detail="LLM processing failed.")

# --- 초기 설정 ---
print("🚀 서버 초기화를 시작합니다...")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = Path("MaterialPalette") / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DB_FOLDER_PATH = "dress-data"
retriever = ColPaliRetriever(db_folder_path=DB_FOLDER_PATH, model_name="tsystems/colqwen2.5-3b-multilingual-v1.0")
llm_orchestrator = LLMOrchestrator(api_key=OPENAI_API_KEY)
app = FastAPI()

# [1. 정적 파일 설정] "static_maps" 폴더를 만들고, /maps 라는 URL 경로로 접근할 수 있게 설정
STATIC_DIR = Path("static_maps")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/maps", StaticFiles(directory=STATIC_DIR), name="maps")

print("✅ 서버 초기화 완료. 요청을 기다립니다.")

@app.get("/")
async def get_test_page():
    return FileResponse("test.html")

@app.post("/apply-style")
async def apply_style(text: str = Form(...), image: UploadFile = File(...)):
    print(f"\n\n--- 새로운 동기식 요청 수신 ---")
    image_basename = Path(image.filename).stem
    session_id = f"{image_basename}_{uuid.uuid4().hex[:8]}"
    pbr_working_dir = OUTPUT_DIR / session_id
    os.makedirs(pbr_working_dir)
    print(f"임시 작업 폴더 생성: {pbr_working_dir}")

    try:
        # 1-4 단계 (이미지 저장, LLM 분석, DB 검색, PBR 파이프라인 실행)
        pattern_image_path = pbr_working_dir / image.filename
        with open(pattern_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print(f"패턴 이미지 저장 완료: {pattern_image_path}")
        parsed_keywords = llm_orchestrator.parse_user_request(text)
        pbr_prompt = parsed_keywords["pbr_prompt"]
        db_query = parsed_keywords["db_query"]
        retrieved_items = retriever.search(query_text=db_query, k=1)
        if not retrieved_items:
            raise HTTPException(status_code=404, detail=f"'{db_query}'에 해당하는 아이템을 DB에서 찾을 수 없습니다.")
        retrieved_item_metadata = retrieved_items[0]['metadata']
        print(f"DB 검색 완료. 타겟 아이템: {retrieved_item_metadata['image_name']}")
        print(f"--- PBR 파이프라인 실행 시작 ---")
        command = ["python", "run_full_pipeline.py", "--image_path", str(pattern_image_path), "--prompt", pbr_prompt]
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        print("--- PBR 파이프라인 실행 완료 ---")
        print("STDOUT:", process.stdout)

        # [2. URL 생성 로직]
        pbr_map_urls = {}
        # Unity에서 접근할 서버의 기본 주소 (실제 환경에 맞게 변경 필요)
        base_url = " https://681929d10a9d.ngrok.app"
        
        # PBR 결과물을 저장할 영구 폴더 (static_maps 내부에 세션 ID로 생성)
        output_static_dir = STATIC_DIR / session_id
        output_static_dir.mkdir(exist_ok=True)

        pbr_map_keywords = {"albedo": "albedo", "normal": "normal", "roughness": "roughness"}
        print("--- 다운로드 URL 생성 시작 ---")
        for map_type, keyword in pbr_map_keywords.items():
            found_files = list(pbr_working_dir.rglob(f'*{keyword}*.png'))
            if found_files:
                source_path = found_files[0]
                dest_path = output_static_dir / source_path.name
                
                # 생성된 PBR 파일을 static 폴더로 복사
                shutil.copy(source_path, dest_path)
                
                # 최종 다운로드 URL 조합
                file_url = f"{base_url}/maps/{session_id}/{source_path.name}"
                pbr_map_urls[map_type] = file_url
                print(f"✅ '{source_path.name}' URL 생성: {file_url}")
            else:
                pbr_map_urls[map_type] = None
                print(f"⚠️ '{keyword}' 키워드를 포함하는 파일을 찾을 수 없음")
        
        # [3. 최종 응답 형식 변경]
        target_filename_stem = Path(retrieved_item_metadata['image_name']).stem
        final_response = {
            "target_filename": target_filename_stem,
            "pbr_map_urls": pbr_map_urls
        }
        return JSONResponse(content=final_response)

    except subprocess.CalledProcessError as e:
        print(f"🚨 PBR 파이프라인 오류 발생: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"PBR 생성 실패: {e.stderr}")
    except Exception as e:
        print(f"🚨 처리 중 오류 발생: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 작업에 사용된 임시 폴더는 정리
        if pbr_working_dir.exists():
            shutil.rmtree(pbr_working_dir)
            print(f"임시 작업 폴더 삭제: {pbr_working_dir}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)