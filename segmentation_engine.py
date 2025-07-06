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
        gd_model_id = "IDEA-Research/grounding-dino-base"
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(self.device)

        print("Loading SAM 2 model...")
        # 이 경로들은 실제 환경에 맞게 확인이 필요합니다.
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
        """
        이미지와 텍스트 프롬프트를 기반으로 마스크를 추출합니다.
        """
        print(f"\nExtracting mask for: '{text_prompt}'...")
        image_np = np.array(image_pil)
        
        # GroundingDINO로 객체 위치 찾기
        inputs = self.gd_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image_pil.size[::-1]]
        )
        
        boxes = results[0]["boxes"]
        if len(boxes) == 0:
            print("Warning: No objects detected for the given prompt.")
            return None

        # SAM2로 마스크 생성
        self.sam2_predictor.set_image(image_np)
        masks, _, _ = self.sam2_predictor.predict(
            box=boxes.to(self.device),
            multimask_output=False
        )
        
        # 마스크 후처리
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        final_mask = np.any(masks, axis=0)
        print("✅ Mask extracted successfully.")
        return final_mask