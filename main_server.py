import os
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from typing import Optional

# 우리가 만든 추론 엔진을 임포트합니다.
from inference_engine import ColPaliRetriever, LLMGenerator, QueryRouter, PatternExtractor

# --- 1. 초기 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_FILE = "artworks_database_andre_500.pt"
COLPALI_MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"
TOP_K = 5

print("Initializing FastAPI server and loading models...")
retriever = ColPaliRetriever(db_path=DATABASE_FILE, model_name=COLPALI_MODEL_NAME)
router = QueryRouter(api_key=OPENAI_API_KEY)
generator = LLMGenerator(api_key=OPENAI_API_KEY)
pattern_extractor = PatternExtractor()
print("✅ Initialization complete. Server is ready.")

app = FastAPI()

# --- 2. API 엔드포인트 정의 ---
@app.post("/query")
async def handle_query(
    query_text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None)
):
    print(f"\nReceived query: text='{query_text}', image='{image.filename if image else None}', k={TOP_K}")

    if not query_text and not image:
        raise HTTPException(status_code=400, detail="No valid query provided.")

    pil_image = None
    if image and image.filename:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")

    # [핵심 로직 수정] AI 라우터를 통해 검색 방식 결정
    # --------------------------------------------------------------------
    search_method = router.decide_search_method(query_text, has_image=(pil_image is not None))

    if search_method == "image_only":
        print("-> Executing Image-only Search based on AI decision.")
        retrieved_items = retriever.search(query_image=pil_image, k=TOP_K)
        llm_query = query_text if query_text else "업로드된 이미지" # LLM에게는 원래 텍스트도 전달
    
    else: # search_method == "text_only"
        print("-> Executing Text-only Search based on AI decision.")
        retrieved_items = retriever.search(query_text=query_text, k=TOP_K)
        llm_query = query_text
        
    # --------------------------------------------------------------------

    # LLM 답변 생성 및 최종 응답 구성
    intro_sentence = generator.generate_intro_sentence(llm_query)
    final_response_items = []
    print("Generating concise descriptions for each retrieved item...")
    
    for i, item in enumerate(retrieved_items):
        print(f"  - Generating description for item {i+1} (Index: {item['metadata'].get('index', 'N/A')}): {item['metadata'].get('title', 'N/A')}")
        llm_description = generator.generate_item_description(item['metadata'])
        final_response_items.append({
            "metadata": item['metadata'],
            "llm_description": f"{i+1}. {llm_description}",
            "image_filename": os.path.basename(item['image_path']) 
        })

    return JSONResponse(content={
        "intro_sentence": intro_sentence,
        "retrieved_results": final_response_items
    })

@app.post("/generate-pattern")
async def handle_pattern_generation(
    image_filename: str = Form(...),
    x: int = Form(...),
    y: int = Form(...)
):
    """유니티에서 선택한 이미지 파일명과 클릭 좌표를 받아 패턴 생성을 요청합니다."""
    print(f"\nReceived pattern generation request for '{image_filename}' at ({x}, {y})")
    
    try:
        # PatternExtractor를 호출하여 실제 로직을 수행합니다.
        result = pattern_extractor.extract_pattern_and_create_pbr(
            image_filename=image_filename, x=x, y=y
        )
        return JSONResponse(content=result)
    except Exception as e:
        print(f"ERROR in pattern generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate pattern.")