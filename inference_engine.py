import os
import torch
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# ColPali 모델 및 프로세서 임포트
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# 스크립트 시작 시 .env 파일을 읽어 환경변수로 로드
load_dotenv()

# ---------------------------------------------
# Class 1: 검색 담당
# ---------------------------------------------
class ColPaliRetriever:
    def __init__(self, db_path, model_name, device="cuda:0"):
        self.device = device
        print(f"Retriever: Loading model '{model_name}'...")
        self.model = ColQwen2_5.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
        print(f"Retriever: Loading database from '{db_path}'...")
        try:
            self.database = torch.load(db_path, map_location='cpu')
            self.db_embeddings = [d['embedding_colpali'] for d in self.database]
            print("Retriever: Database loaded successfully.")
        except FileNotFoundError:
            self.database = None
            self.db_embeddings = []
            print(f"Retriever ERROR: Database file not found at '{db_path}'")

    def search(self, query_text=None, query_image=None, k=5):
        """[수정됨] 이 함수는 이제 '어떻게' 검색할지 고민하지 않고, 주어진 인자로 '단순 검색'만 수행합니다."""
        if self.database is None:
            return []

        with torch.no_grad():
            # 이미지만으로 검색하거나
            if query_image:
                inputs = self.processor.process_images(images=[query_image]).to(self.device)
            # 텍스트만으로 검색합니다.
            elif query_text:
                inputs = self.processor.process_queries(queries=[query_text]).to(self.device)
            else:
                return []
            
            query_embed = list(torch.unbind(self.model(**inputs).cpu()))
            scores = self.processor.score(query_embed, self.db_embeddings, device=self.device)
            indices = scores[0].topk(k).indices.tolist()
            results = [self.database[i] for i in indices]
            return results

# ---------------------------------------------
# Class 2: LLM 쿼리 라우터 (역할 명확)
# ---------------------------------------------
class QueryRouter:
    def __init__(self, api_key, model="gpt-4o-mini"):
        if not api_key or not api_key.startswith("sk-"):
            self.client = None
            return
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def decide_search_method(self, query_text: str, has_image: bool):
        if self.client is None or not query_text:
            return "image_only" if has_image else "text_only"
        if not has_image:
            return "text_only"

        prompt = f"""당신은 사용자의 의도를 파악하는 AI 어시스턴트입니다. 사용자가 이미지와 함께 텍스트로 질문했습니다. 텍스트의 내용을 보고, 이 요청을 처리하기 위해 '이미지 검색'과 '텍스트 검색' 중 어느 것이 더 적합한지 결정해주세요.
- "이것", "이 그림", "이 패턴", "이 스타일" 등 이미지를 직접적으로 지칭하는 말이 포함되어 있다면, 사용자의 주된 의도는 이미지에 있으므로 'image_only'라고 답변해야 합니다.
- "용 무늬", "샤넬 스타일", "1950년대 드레스" 등 이미지와는 별개의 구체적인 개념이나 속성을 텍스트로 설명하고 있다면, 텍스트가 더 중요하므로 'text_only'라고 답변해야 합니다.
사용자 텍스트 쿼리: "{query_text}"
결정 (오직 'image_only' 또는 'text_only' 둘 중 하나로만 답변):"""

        try:
            print(f"-> Routing query: '{query_text}'")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intelligent dispatcher that analyzes a user's query to decide the search method. You must respond with only 'image_only' or 'text_only'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, max_tokens=5,
            )
            decision = response.choices[0].message.content.strip().lower()
            if decision in ["image_only", "text_only"]:
                print(f"-> AI Decision: {decision}")
                return decision
            else:
                print(f"-> AI Decision failed, defaulting to text_only. (Response: {decision})")
                return "text_only" 
        except Exception as e:
            print(f"QueryRouter Error: {e}. Defaulting to text_only.")
            return "text_only"

# ---------------------------------------------
# Class 3: LLM 답변 생성 담당
# ---------------------------------------------
class LLMGenerator:
    def __init__(self, api_key, model="gpt-4o-mini"):
        if not api_key or not api_key.startswith("sk-"):
            self.client = None
            print("Warning: OpenAI API key not found or invalid. LLMGenerator will be disabled.")
            return
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_intro_sentence(self, user_query: str):
        """사용자의 쿼리를 받아, 검색 결과를 소개하는 안내 문장 한 줄을 생성합니다."""
        if self.client is None: return "검색 결과입니다."

        prompt = f"""사용자의 검색 쿼리를 분석해서, 검색 결과를 소개하는 짧고 친절한 안내 문장을 딱 한 문장으로 만들어줘.
- 사용자가 영어로 검색했다면, '영어(한글)' 형식으로 키워드를 알려줘. 예: 'dragon' -> '용(dragon)'
- 사용자가 한글로 검색했다면, 그냥 그 키워드를 사용해. 예: '샤넬 수트' -> '샤넬 수트'
- 최종 결과물은 "OOO 키워드로 검색하셨군요. 관련된 아이템들을 소개해 드립니다." 형식으로 만들어줘.

사용자 쿼리: "{user_query}"
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates a polite introductory sentence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM intro generation Error: {e}")
            return "검색 결과입니다."


    def generate_item_description(self, item_metadata: dict):
        """개별 아이템의 메타데이터를 받아 100자 내외의 사실 기반 요약을 생성합니다."""
        if self.client is None: return item_metadata.get('description', '설명이 없습니다.')[:100]

        context = f"제목: {item_metadata.get('title', '')}\n설명: {item_metadata.get('description', '')}"

        prompt = f"""당신은 전문 아카비스트입니다. 아래에 주어진 '참고 정보'의 내용에만 근거해서, 이 패션 아이템에 대한 핵심 요약 설명을 100자 이내의 간결한 한국어 문장으로 작성해주세요.
- 절대로 참고 정보에 없는 내용을 상상하거나 추측해서 추가하지 마세요.
- 사실만을 객관적으로 전달해야 합니다.

### 참고 정보:
{context}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional archivist who summarizes fashion items factually and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM item description Error: {e}")
            return "설명을 생성하는 중 오류가 발생했습니다."