# inference_engine.py

import os
import torch
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

# ColPali 모델 및 프로세서 임포트
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# ---------------------------------------------
# Class 1: 검색 담당 (최종 수정)
# ---------------------------------------------
class ColPaliRetriever:
    def __init__(self, db_folder_path, model_name, device="cuda:0"):
        self.device = device
        self.db_folder_path = Path(db_folder_path)
        
        print(f"Retriever: Loading model '{model_name}'...")
        self.model = ColQwen2_5.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
        
        # 폴더로부터 데이터베이스 빌드 (메타데이터 없이)
        self.database, self.db_embeddings = self._build_database_from_folder()

    def _build_database_from_folder(self, batch_size=8): # 배치 크기를 인자로 추가 (GPU 사양에 맞게 조절)
        """지정된 폴더의 이미지를 읽어 DB와 이미지 임베딩을 생성합니다."""
        if not self.db_folder_path.exists():
            print(f"Retriever ERROR: Database folder not found at '{self.db_folder_path}'")
            return [], []

        print(f"Retriever: Building database from image folder '{self.db_folder_path}'...")
        database = []
        db_embeddings = [] # 최종 임베딩을 담을 리스트
        
        supported_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [p for p in self.db_folder_path.iterdir() if p.suffix.lower() in supported_extensions]

        print(f"Retriever: Found {len(image_files)} images. Now generating embeddings in batches of {batch_size}...")

        # tqdm으로 전체 진행 상황 표시
        with tqdm(total=len(image_files), desc="Generating Embeddings") as pbar:
            # 전체 이미지 파일을 배치 크기만큼 나누어 처리
            for i in range(0, len(image_files), batch_size):
                # 현재 처리할 배치(이미지 파일 경로 묶음)
                batch_paths = image_files[i:i+batch_size]
                batch_images = [] # 실제 이미지 데이터를 담을 리스트

                for image_path in batch_paths:
                    try:
                        pil_image = Image.open(image_path).convert("RGB")
                        item = {
                            'image_name': image_path.name,
                            'image_path': str(image_path),
                            'file_number': image_path.stem.split('_')[0],
                            'pil_image': pil_image
                        }
                        database.append(item)
                        batch_images.append(pil_image)
                    except Exception as e:
                        print(f"\nWarning: Failed to process {image_path}. Error: {e}")
                
                # 현재 배치에 이미지가 있을 경우에만 임베딩 생성
                if batch_images:
                    with torch.no_grad():
                        inputs = self.processor.process_images(images=batch_images).to(self.device)
                        # 생성된 임베딩을 최종 리스트에 추가
                        batch_embeddings = list(torch.unbind(self.model(**inputs).cpu()))
                        db_embeddings.extend(batch_embeddings)
                
                pbar.update(len(batch_paths)) # 진행 바 업데이트
        
        print("\nRetriever: Database built and image embeddings generated successfully.")
        return database, db_embeddings

    def search(self, query_text=None, k=1):
        """텍스트 쿼리로 이미지 DB를 검색합니다."""
        if not self.database or not query_text:
            return []

        with torch.no_grad():
            # 텍스트 쿼리를 임베딩으로 변환
            inputs = self.processor.process_queries(queries=[query_text]).to(self.device)
            query_embed = list(torch.unbind(self.model(**inputs).cpu()))
            
            # 텍스트 쿼리 임베딩과 이미지 DB 임베딩 간의 유사도 점수 계산
            scores = self.processor.score(query_embed, self.db_embeddings, device=self.device)
            indices = scores[0].topk(k).indices.tolist()
            
            # 검색 결과에 실제 DB 아이템 정보를 매칭
            results = [{'metadata': self.database[i]} for i in indices]
            return results