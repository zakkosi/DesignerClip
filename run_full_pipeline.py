# run_full_pipeline.py (최종 수정본)

import os
import shutil
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Full pipeline from mask generation to PBR texture creation.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the source image file inside MaterialPalette/output/<id>/')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for mask generation.')
    args = parser.parse_args()

    # --- 새로운 폴더 구조에 맞게 경로 재설정 ---
    source_image_path = Path(args.image_path)
    if not source_image_path.exists():
        print(f"오류: 원본 이미지 '{source_image_path}'를 찾을 수 없습니다.")
        return

    # 작업 폴더는 전달받은 이미지의 부모 폴더입니다. (예: MaterialPalette/output/00000)
    working_dir = source_image_path.parent
    image_basename = source_image_path.stem # 예: '00000'
    
    # MaterialPalette 폴더의 경로는 working_dir의 조부모 폴더가 됩니다.
    material_palette_dir = working_dir.parent.parent # 예: MaterialPalette
    
    # 마스크 파일 이름 및 저장 경로 정의
    mask_filename = f"{image_basename}_mask.png"
    masks_dir = working_dir / "masks"

    print("="*50)
    print("🚀 전체 파이프라인을 시작합니다. (최종 로직)")
    print(f"- 원본 이미지: {source_image_path}")
    print(f"- 작업 폴더: {working_dir}")
    print("="*50)

    try:
        # --- 1단계: 마스크 생성 ---
        print("\n[1/3] 마스크 생성을 시작합니다...")
        mask_gen_command = [
            "python", "main_test.py",
            "--image_path", str(source_image_path),
            "--prompt", args.prompt,
            "--output", mask_filename
        ]
        subprocess.run(mask_gen_command, check=True)
        print(f"✅ 마스크 생성 완료: {mask_filename}")

        # --- 2단계: masks 폴더 생성 및 마스크 파일 이동 ---
        print("\n[2/3] masks 폴더를 구성합니다...")
        os.makedirs(masks_dir, exist_ok=True)
        
        # 생성된 마스크를 masks 폴더로 이동
        shutil.move(mask_filename, masks_dir)
        print(f"  - 마스크 파일 이동 완료: {masks_dir / mask_filename}")
        print("✅ 폴더 구성 완료.")

        # --- 3단계: MaterialPalette 파이프라인 실행 ---
        print("\n[3/3] MaterialPalette PBR 생성을 시작합니다...")
        
        # pipeline.py에 전달할 인자는 'output/00000' 형태가 되어야 합니다.
        pipeline_target_path = working_dir.relative_to(material_palette_dir)
        pbr_gen_command = ["python", "pipeline.py", str(pipeline_target_path)]
        
        # MaterialPalette 폴더에서 명령어를 실행
        subprocess.run(pbr_gen_command, check=True, cwd=material_palette_dir)
        
        print("\n🎉 모든 파이프라인 작업이 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"  - 명령어: {e.cmd}")
            print(f"  - 종료 코드: {e.returncode}")
            print(f"  - Stderr: {e.stderr}")

if __name__ == "__main__":
    main()