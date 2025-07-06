import argparse
from PIL import Image
import numpy as np
from segmentation_engine import SegmentationEngine

def main(args):
    print("--- Initializing Segmentation Engine ---")
    try:
        seg_engine = SegmentationEngine()
    except Exception as e:
        print(f"Error initializing SegmentationEngine: {e}")
        print("Please ensure you are in the correct conda environment with Grounded-SAM2 installed.")
        return

    print(f"--- Loading image: {args.input_path} ---")
    try:
        image_pil = Image.open(args.input_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {args.input_path}")
        return

    # 1. 의류 마스크 추출
    clothing_mask = seg_engine.extract_mask(image_pil, text_prompt="dress . clothing .")

    if clothing_mask is None:
        print("Mask could not be detected.")
        return

    # 2. 흑백 마스크 이미지 파일로 저장
    mask_image_np = (clothing_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_image_np)

    mask_pil.save(args.output_path)
    print(f"✅ Mask file successfully saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a binary mask from an image.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output mask PNG file.')
    args = parser.parse_args()
    main(args)