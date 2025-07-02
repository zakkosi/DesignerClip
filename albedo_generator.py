import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline

class AlbedoGenerator:
    def __init__(self, device="cuda:0"):
        self.device = device
        print("--- Initializing AlbedoGenerator ---")
        
        # Inpainting 파이프라인 로드 (초기화 시 한 번만 실행)
        model_path = "stabilityai/stable-diffusion-2-inpainting"
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        print("✅ AlbedoGenerator is ready.")

    def generate_for_local_pattern(self, 
                                   original_image: Image.Image, 
                                   feature_mask: np.ndarray,
                                   prompt: str = "photorealistic plain fabric texture, seamless"):
        """
        Local Pattern에 대한 Albedo 맵을 생성합니다.
        """
        print("\nGenerating Albedo map for LOCAL pattern...")
        
        # 1. Pillow와 호환되도록 마스크 이미지 생성 (지워야 할 부분이 흰색)
        mask_pil = Image.fromarray((feature_mask * 255).astype(np.uint8))

        # 2. Diffusion Inpainting 실행하여 '기본 옷감' 생성
        #    - 원본 이미지, 지울 마스크, 프롬프트를 입력
        print("  - Step 1: Generating base fabric texture with inpainting...")
        base_fabric_image = self.inpaint_pipe(
            prompt=prompt,
            image=original_image.resize((512, 512)), # 모델 입력 사이즈에 맞게 조절
            mask_image=mask_pil.resize((512, 512)),
            strength=1.0 # 마스크 영역을 완전히 새로 생성
        ).images[0]
        
        base_fabric_image = base_fabric_image.resize(original_image.size) # 원본 사이즈로 복원

        # 3. 원본에서 '핵심 특징' 텍스처 추출
        print("  - Step 2: Extracting original feature texture...")
        original_np = np.array(original_image)
        feature_texture_np = np.zeros_like(original_np)
        feature_texture_np[feature_mask] = original_np[feature_mask]

        # 4. '기본 옷감'과 '핵심 특징' 합성
        print("  - Step 3: Compositing textures...")
        final_albedo_np = np.array(base_fabric_image)
        # feature_mask가 True인 영역에만 feature_texture_np 값을 덮어쓰기
        final_albedo_np[feature_mask] = feature_texture_np[feature_mask]
        
        final_albedo_pil = Image.fromarray(final_albedo_np)
        
        print("✅ Albedo map generation complete.")
        return final_albedo_pil

    def generate_for_global_pattern(self, original_image: Image.Image, clothing_mask: np.ndarray):
        """
        Global Pattern에 대한 Albedo 맵을 생성합니다. (타일링 가능한 패치 추출 등)
        """
        print("\nGenerating Albedo map for GLOBAL pattern...")
        # TODO: 옷 마스크 영역에서 대표적인 패턴 패치를 잘라내어 반환하는 로직 구현
        # 우선은 마스크된 전체 이미지를 반환하는 것으로 간단히 구현
        original_np = np.array(original_image)
        masked_image = np.zeros_like(original_np)
        masked_image[clothing_mask] = original_np[clothing_mask]
        
        print("✅ Albedo map (masked original) generation complete.")
        return Image.fromarray(masked_image)