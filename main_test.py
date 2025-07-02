import os
import csv
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np

from segmentation_engine import SegmentationEngine
from analysis_engine import PatternAnalyzer

def process_single_image(args, seg_engine, analysis_engine):
    print(f"--- Processing single image: {args.image_path} ---")
    try:
        image_pil = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {args.image_path}")
        return

    clothing_mask = seg_engine.extract_mask(image_pil, text_prompt="dress . clothing .")
    if clothing_mask is None:
        print("Could not process image further as no mask was detected.")
        return

    image_np = np.array(image_pil)
    # [수정] args에서 받은 하이퍼파라미터를 엔진에 전달
    pattern_type, score = analysis_engine.analyze_consistency(
        pattern_image_np=image_np, 
        mask=clothing_mask,
        target_patches=args.target_patches,
        min_patches=args.min_patches,
        consistency_threshold=args.threshold
    )

    print("\n--- FINAL RESULT ---")
    print(f"Image: {args.image_path}")
    print(f"Detected Pattern Type: {pattern_type.upper()}")
    print(f"Consistency Score (Std. Dev): {score:.4f}")

    if pattern_type == 'local':
        feature_mask = seg_engine.extract_feature_from_mask(
            image_pil, 
            clothing_mask, 
            feature_prompt="gold embroidery . metallic ornament", 
        )
        
        if feature_mask is not None:
            # 시각적 확인을 위해 자수 부분만 잘라낸 이미지 저장
            feature_image_np = np.zeros_like(image_np)
            feature_image_np[feature_mask] = image_np[feature_mask]
            Image.fromarray(feature_image_np).save("extracted_feature.png")
            print("✅ Saved the extracted feature to 'extracted_feature.png'")


def process_batch(args, seg_engine, analysis_engine):
    print(f"--- Processing batch from: {args.input_dir} ---")
    results = []
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the input directory.")
        return

    for filename in tqdm(image_files, desc="Analyzing Images"):
        image_path = os.path.join(args.input_dir, filename)
        try:
            image_pil = Image.open(image_path).convert("RGB")
            clothing_mask = seg_engine.extract_mask(image_pil, text_prompt="dress . clothing .")
            if clothing_mask is None:
                results.append([filename, 'mask_detection_failed', -1.0])
                continue

            image_np = np.array(image_pil)
            # [수정] args에서 받은 하이퍼파라미터를 엔진에 전달
            pattern_type, score = analysis_engine.analyze_consistency(
                pattern_image_np=image_np, 
                mask=clothing_mask,
                target_patches=args.target_patches,
                min_patches=args.min_patches,
                consistency_threshold=args.threshold
            )
            results.append([filename, pattern_type, f"{score:.4f}"])
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append([filename, 'processing_error', -1.0])

    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'pattern_type', 'consistency_score'])
            writer.writerows(results)
        print(f"\n--- BATCH COMPLETE ---")
        print(f"✅ Results saved to {args.output_csv}")
    except Exception as e:
        print(f"\nError saving CSV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pattern Analysis Pipeline")
    # [수정] 공통 인자들을 부모 파서에 추가
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--target_patches', type=int, default=16, help='Number of patches to attempt to sample.')
    parent_parser.add_argument('--min_patches', type=int, default=12, help='Minimum valid patches required for analysis.')
    parent_parser.add_argument('--threshold', type=float, default=0.05, help='Standard deviation threshold for consistency.')
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Processing mode: single or batch')

    parser_single = subparsers.add_parser('single', help='Process a single image', parents=[parent_parser])
    parser_single.add_argument('--image_path', type=str, required=True, help='Path to the single image file')

    parser_batch = subparsers.add_parser('batch', help='Process a directory of images', parents=[parent_parser])
    parser_batch.add_argument('--input_dir', type=str, required=True, help='Directory containing images')
    parser_batch.add_argument('--output_csv', type=str, default='analysis_results.csv', help='Path to save the output CSV file')
    
    args = parser.parse_args()

    segmentation_engine = SegmentationEngine()
    analysis_engine = PatternAnalyzer()

    if args.mode == 'single':
        process_single_image(args, segmentation_engine, analysis_engine)
    elif args.mode == 'batch':
        # 배치 모드는 일단 기능이 복잡해지므로 나중에 추가
        print("Batch mode currently does not support 2nd-pass feature extraction.")
        pass