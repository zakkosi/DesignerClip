import torch
import open_clip
from PIL import Image
import numpy as np

class PatternAnalyzer:
    def __init__(self, device="cuda:0"):
        # ... __init__ 부분은 변경 없음 ...
        self.device = device
        print("--- Initializing PatternAnalyzer (Upgraded) ---")
        print("Loading OpenCLIP model...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        print("✅ PatternAnalyzer is ready.")

    def _get_random_patches_with_retry(self, image_np: np.ndarray, mask: np.ndarray, 
                                     target_patches: int, patch_size=64, max_retries=5):
        # [수정] target_patches를 인자로 받음
        valid_points = np.argwhere(mask)
        if len(valid_points) == 0:
            return []

        patches = []
        attempt = 0
        half_size = patch_size // 2
        
        while len(patches) < target_patches and attempt < max_retries:
            needed = target_patches - len(patches)
            selected_indices = np.random.choice(len(valid_points), needed)
            centers = valid_points[selected_indices]
            
            for y, x in centers:
                if y - half_size >= 0 and y + half_size <= image_np.shape[0] and \
                   x - half_size >= 0 and x + half_size <= image_np.shape[1]:
                    patch_np = image_np[y - half_size : y + half_size, x - half_size : x + half_size]
                    patches.append(Image.fromarray(patch_np))
            attempt += 1
        
        return patches

    def analyze_consistency(self, pattern_image_np: np.ndarray, mask: np.ndarray, 
                          target_patches: int = 16, min_patches: int = 12, consistency_threshold: float = 0.05):
        # [수정] 모든 하이퍼파라미터를 인자로 받도록 변경
        print("\nAnalyzing pattern consistency (Upgraded Method)...")
        
        patches = self._get_random_patches_with_retry(pattern_image_np, mask, target_patches=target_patches)
        
        if len(patches) < min_patches:
            print(f"  - Not enough valid patches ({len(patches)}/{min_patches}). Classified as LOCAL.")
            return "local", -1.0

        print(f"  - Extracted {len(patches)} valid patches.")

        with torch.no_grad():
            preprocessed_patches = [self.clip_preprocess(p) for p in patches]
            batch_tensor = torch.stack(preprocessed_patches).to(self.device)
            
            print(f"  - Processing {len(patches)} patches in a single batch...")
            patch_vectors = self.clip_model.encode_image(batch_tensor)
            patch_vectors /= patch_vectors.norm(dim=-1, keepdim=True)

        sim_matrix = patch_vectors @ patch_vectors.T
        off_diagonal_indices = ~torch.eye(len(patches), dtype=bool)
        similarity_values = sim_matrix[off_diagonal_indices]
        std_dev = torch.std(similarity_values).item()
        
        print(f"  - Similarity Std. Deviation: {std_dev:.4f}")
        
        if std_dev < consistency_threshold:
            print(f"  - Result: Low deviation. Classified as GLOBAL.")
            return "global", std_dev
        else:
            print(f"  - Result: High deviation. Classified as LOCAL.")
            return "local", std_dev