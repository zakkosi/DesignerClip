import argparse
from PIL import Image
import numpy as np
from segmentation_engine import SegmentationEngine

def main():
    # 1. 스크립트 실행 시 필요한 인자(argument) 설정
    parser = argparse.ArgumentParser(description="Extract a mask from an image using a text prompt.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--prompt', type=str, default="dress . clothing .", help='Text prompt to describe the object to mask.')
    parser.add_argument('--output', type=str, default="output_mask.png", help='Path to save the output mask image.')
    args = parser.parse_args()

    print(f"--- Starting Mask Generation ---")
    print(f"  - Image: {args.image_path}")
    print(f"  - Prompt: '{args.prompt}'")
    print(f"  - Output: {args.output}")
    
    # 2. 모델 로드
    try:
        seg_engine = SegmentationEngine()
    except Exception as e:
        print(f"\nError initializing SegmentationEngine: {e}")
        print("Please check model paths and dependencies.")
        return

    # 3. 이미지 열기
    try:
        image_pil = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"\nError: Input image not found at '{args.image_path}'")
        return

    # 4. 마스크 추출 실행
    clothing_mask = seg_engine.extract_mask(image_pil, text_prompt=args.prompt)
    
    # 5. 결과 저장
    if clothing_mask is not None:
        # boolean 마스크(True/False)를 흑백 이미지(0/255)로 변환
        mask_image_np = (clothing_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_image_np)
        
        # 파일로 저장
        mask_pil.save(args.output)
        print(f"\n✅ Mask file saved successfully to '{args.output}'")
    else:
        print("\nCould not save mask because no objects were detected.")

if __name__ == "__main__":
    main()