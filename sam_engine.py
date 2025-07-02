import torch
import open_clip
from PIL import Image
import cv2
import numpy as np
import os
import sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 이 스크립트 파일(sam_engine.py)이 있는 폴더의 절대 경로를 가져옵니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Grounded-SAM-2 폴더의 절대 경로를 만듭니다.
GSAM_PATH = os.path.join(SCRIPT_DIR, "Grounded-SAM-2")
# Grounded-SAM-2 폴더 내부의 모듈을 임포트할 수 있도록 경로를 추가합니다.
sys.path.append(GSAM_PATH)

class PatternEngine:
    def __init__(self, device="cuda:0"):
        self.device = device
        print("--- Initializing New PatternEngine with Grounded-SAM 2 ---")

        # --- Grounding DINO 모델 로딩 ---
        print("Loading Grounding DINO model...")
        gd_model_id = "IDEA-Research/grounding-dino-tiny"
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(self.device)

        # --- SAM 2 모델 로딩 (핵심 수정 부분) ---
        print("Loading SAM 2 model...")
        # 체크포인트는 절대 경로로 지정 (torch.load는 절대 경로를 잘 처리함)
        sam2_checkpoint = os.path.join(GSAM_PATH, "checkpoints/sam2.1_hiera_large.pt")
        # 설정 파일은 Hydra를 위해 '상대 경로' 이름으로 지정
        sam2_model_config_relative = "configs/sam2.1/sam2.1_hiera_l.yaml"

        if not os.path.exists(sam2_checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found at '{sam2_checkpoint}'. Please run 'bash download_ckpts.sh' in the 'Grounded-SAM-2/checkpoints' directory.")

        # Hydra가 올바르게 작동하도록, 작업 디렉토리를 잠시 변경했다가 복원합니다.
        current_dir = os.getcwd()
        try:
            os.chdir(GSAM_PATH) # 작업 디렉토리를 Grounded-SAM-2로 변경
            # Hydra가 기대하는 상대 경로 이름으로 모델을 빌드합니다.
            sam2_model = build_sam2(sam2_model_config_relative, sam2_checkpoint, device=self.device)
        finally:
            os.chdir(current_dir) # 원래 디렉토리로 안전하게 복귀
        
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # --- CLIP 모델 로딩 ---
        print("Loading OpenCLIP model...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        print("✅ All models loaded. Engine is ready.")

    def extract_clothing_mask(self, image_path: str, text_prompt: str = "dress . clothing ."):
        # (내부 로직은 이전과 동일하므로 변경 없음)
        try:
            print(f"\nStep 1: Extracting '{text_prompt}' from '{image_path}'...")
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)

            inputs = self.gd_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gd_model(**inputs)
            
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=[image_pil.size[::-1]]
            )
            
            input_boxes = results[0]["boxes"]
            if input_boxes.dim() == 0 or input_boxes.size(0) == 0:
                print("Warning: No objects detected by Grounding DINO.")
                return None, None

            self.sam2_predictor.set_image(image_np)
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes.to(self.device),
                multimask_output=False,
            )
            
            if masks.ndim ==4:
                masks = masks.squeeze(1)

            final_mask = np.any(masks, axis=0)
            
            print("✅ Clothing mask extracted successfully.")
            return final_mask, image_np

        except Exception as e:
            print(f"Error during clothing extraction: {e}")
            return None, None

# --- 테스트를 위한 임시 실행 코드 ---
if __name__ == '__main__':
    TEST_IMAGE_PATH = "TEST2.png"
    TEXT_PROMPT = "dress . accessories ."
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[오류] 테스트 이미지를 찾을 수 없습니다: '{TEST_IMAGE_PATH}'")
    else:
        engine = PatternEngine()
        clothing_mask, original_image = engine.extract_clothing_mask(
            image_path=TEST_IMAGE_PATH,
            text_prompt=TEXT_PROMPT
        )

        if clothing_mask is not None:
            output_image_np = np.zeros_like(original_image)
            output_image_np[clothing_mask] = original_image[clothing_mask]
            
            output_image_pil = Image.fromarray(output_image_np)
            output_filename = "detected_clothing_mask.jpg"
            output_image_pil.save(output_filename)
            
            print(f"\n--- 최종 결과 ---")
            print(f"✅ Test successful! Result saved to '{output_filename}'")
            print("이미지 파일을 열어서 옷 부분만 제대로 추출되었는지 확인해보세요.")
        else:
            print("\n--- 최종 결과 ---")
            print("❌ Test failed. Could not extract clothing mask.")