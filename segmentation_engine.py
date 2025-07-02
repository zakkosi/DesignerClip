import torch
import numpy as np
from PIL import Image
import os, sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GSAM_PATH = os.path.join(SCRIPT_DIR, "Grounded-SAM-2")
sys.path.append(GSAM_PATH)


class SegmentationEngine:
    def __init__(self, device="cuda:0"):
        self.device = device
        print("--- Initializing SegmentationEngine ---")

        print("Loading Grounding DINO model...")
        gd_model_id = "IDEA-Research/grounding-dino-base" # base 모델 사용
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(self.device)

        print("Loading SAM 2 model...")
        sam2_checkpoint = os.path.join(GSAM_PATH, "checkpoints/sam2.1_hiera_large.pt")
        sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        current_dir = os.getcwd()
        try:
            os.chdir(GSAM_PATH)
            sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
        finally:
            os.chdir(current_dir)
        
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("✅ SegmentationEngine is ready.")

    def extract_mask(self, image_pil: Image, text_prompt: str):
        print(f"\nExtracting '{text_prompt}'...")
        image_np = np.array(image_pil)
        
        inputs = self.gd_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image_pil.size[::-1]]
        )
        
        if len(results[0]["boxes"]) == 0:
            print("Warning: No objects detected.")
            return None

        self.sam2_predictor.set_image(image_np)
        masks, _, _ = self.sam2_predictor.predict(
            box=results[0]["boxes"].to(self.device), multimask_output=False
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        final_mask = np.any(masks, axis=0)
        print("✅ Mask extracted.")
        return final_mask
    
    def extract_feature_from_mask(self, image_pil: Image, clothing_mask: np.ndarray, feature_prompt: str,
                              box_threshold: float = 0.3, # 2차 탐지 임계값을 좀 더 높게 설정
                              max_box_area_ratio: float = 0.8): # 원본 마스크 대비 최대 박스 크기 비율
        """
        1차 마스크(옷) 영역 내에서, 2차 프롬프트(자수)에 해당하는 특징을 추출합니다.
        """
        print(f"\nExtracting sub-feature '{feature_prompt}' from within the clothing mask...")
        image_np = np.array(image_pil)

        # 1. 1차 마스크 영역 밖을 검게 칠해서, 탐색 범위를 옷 내부로 제한합니다.
        masked_image_np = image_np.copy()
        masked_image_np[~clothing_mask] = 0
        masked_image_pil = Image.fromarray(masked_image_np)

        # 2. Grounded-DINO를 이 '마스킹된 이미지'에 대해 실행합니다.
        inputs = self.gd_processor(images=masked_image_pil, text=feature_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        
        # [수정] 후처리 시 box_threshold를 인자로 받은 값으로 사용합니다.
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=box_threshold, target_sizes=[masked_image_pil.size[::-1]]
        )
        
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        
        if len(boxes) == 0:
            print("Warning: No sub-features detected.")
            return None
        
        # --- ✨ 핵심 필터링 로직 추가 ✨ ---
        
        # 1. 원본 옷 마스크의 전체 면적 계산
        clothing_mask_area = np.sum(clothing_mask)
        
        # 2. 조건을 만족하는 박스만 필터링
        filtered_boxes = []
        for box, score in zip(boxes, scores):
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            
            # 조건: 박스 크기가 전체 옷 마스크 면적의 일정 비율(예: 80%) 미만이어야 함
            if box_area < clothing_mask_area * max_box_area_ratio:
                filtered_boxes.append(box)
            else:
                print(f"  - Info: Filtering out a large box (area: {box_area:.0f}, score: {score:.2f})")

        if not filtered_boxes:
            print("Warning: All detected boxes were filtered out as too large.")
            return None
            
        filtered_boxes_tensor = torch.stack(filtered_boxes).to(self.device)
        
        # ------------------------------------

        # SAM은 원본 이미지에 대해 마스크를 생성해야 하므로, 원본 이미지로 다시 설정
        self.sam2_predictor.set_image(image_np) 
        
        # [수정] 필터링된 박스만 SAM에 전달
        masks, _, _ = self.sam2_predictor.predict(
            box=filtered_boxes_tensor, 
            multimask_output=False
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
                
        feature_mask = np.any(masks, axis=0)
        print("✅ Sub-feature mask extracted.")
        return feature_mask