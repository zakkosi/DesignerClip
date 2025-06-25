import gradio as gr
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import AutoProcessor, AutoModel

# --- 1. 설정 및 모델 로딩 (이전과 동일) ---
DATABASE_FILE = "artworks_database_andre_500.pt"

#COLPALI_MODEL_NAME = "vidore/colqwen2-v1.0"
COLPALI_MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FASHION_CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
SIGLIP_MODEL_NAME = "google/siglip2-base-patch32-256"

print("="*50)
print("Loading all 4 models for querying. This may take a while...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("="*50)

#colpali_model = ColQwen2.from_pretrained(COLPALI_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device).eval()
#colpali_processor = ColQwen2Processor.from_pretrained(COLPALI_MODEL_NAME)

#
colpali_model = ColQwen2_5.from_pretrained(COLPALI_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device).eval()
colpali_processor = ColQwen2_5_Processor.from_pretrained(COLPALI_MODEL_NAME)

clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

fashion_clip_model = AutoModel.from_pretrained(FASHION_CLIP_MODEL_NAME).to(device).eval()
fashion_clip_processor = AutoProcessor.from_pretrained(FASHION_CLIP_MODEL_NAME)

siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(device).eval()
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

print("\n✅ All models for querying loaded successfully.")

# --- 2. 데이터베이스 로드 및 준비 (수정된 부분) ---
print(f"Loading pre-computed database from '{DATABASE_FILE}'...")
try:
    database = torch.load(DATABASE_FILE, map_location='cpu')
    
    db_colpali_embeds_list = [d['embedding_colpali'] for d in database]
    db_clip_embeds = torch.stack([d['embedding_clip'] for d in database]).to(device)
    db_fashion_clip_embeds = torch.stack([d['embedding_fashion_clip'] for d in database]).to(device)
    db_siglip_embeds = torch.stack([d['embedding_siglip'] for d in database]).to(device)
    
    # [수정됨] 경로뿐만 아니라 실제 PIL 이미지 객체를 미리 모두 로드합니다.
    image_paths = [d['image_path'] for d in database]
    all_images_pil = [Image.open(p) for p in image_paths]
    
    print(f"✅ Database loaded and moved to device '{device}' successfully with {len(database)} entries.")
except FileNotFoundError:
    print(f"❌ ERROR: Database file not found! Please run 'create_database.py' first.")
    database = None

# --- 3. 통합 검색 함수 (수정된 부분) ---
def search_all(query_text, query_image, k):
    if not database:
        error_msg = "Database not loaded."
        return [], error_msg, [], error_msg, [], error_msg, [], error_msg

    if not query_text and not query_image:
        error_msg = "Please enter a text query or upload an image."
        return [], error_msg, [], error_msg, [], error_msg, [], error_msg

    k = min(k, len(database))
    results = {}

    with torch.no_grad():
        # 1. ColPali 검색
        if query_text:
            colpali_query_input = colpali_processor.process_queries([query_text]).to(device)
        else:
            colpali_query_input = colpali_processor.process_images([query_image]).to(device)
        colpali_query_embed = list(torch.unbind(colpali_model(**colpali_query_input).cpu()))
        colpali_scores = colpali_processor.score(colpali_query_embed, db_colpali_embeds_list, device=device)
        colpali_indices = colpali_scores[0].topk(k).indices.tolist()
        # [수정됨] 경로 대신 PIL 이미지 객체를 반환합니다.
        results['colpali'] = [(all_images_pil[i], database[i]['metadata'].get('title', f'Index {i}')) for i in colpali_indices]

        # 2. 나머지 모델 검색
        models_to_search = {
            "clip": (clip_model, clip_processor, db_clip_embeds),
            "fashion_clip": (fashion_clip_model, fashion_clip_processor, db_fashion_clip_embeds),
            "siglip": (siglip_model, siglip_processor, db_siglip_embeds)
        }
        for name, (model, processor, db_embeds) in models_to_search.items():
            if query_text:
                inputs = processor(
                    text=[query_text], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                ).to(device)
                query_embed = model.get_text_features(**inputs)
            else:
                inputs = processor(images=[query_image], return_tensors="pt").to(device)
                query_embed = model.get_image_features(**inputs)
            
            query_embed_norm = query_embed / query_embed.norm(dim=-1, keepdim=True)
            db_embeds_norm = db_embeds / db_embeds.norm(dim=-1, keepdim=True)
            similarity = (query_embed_norm @ db_embeds_norm.T).squeeze(0)
            
            indices = similarity.topk(k).indices.tolist()
            # [수정됨] 경로 대신 PIL 이미지 객체를 반환합니다.
            results[name] = [(all_images_pil[i], database[i]['metadata'].get('title', f'Index {i}')) for i in indices]

    return (
        results['colpali'], "Results from ColPali",
        results['clip'], "Results from CLIP",
        results['fashion_clip'], "Results from Fashion-CLIP",
        results['siglip'], "Results from SigLIP 2"
    )

# --- 4. Gradio UI 구성 (이전과 동일) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Model Image Retrieval Comparison")
    gr.Markdown("Enter a text query or upload an image to compare the retrieval results of four different AI models side-by-side.")

    with gr.Row():
        status = "Ready to search." if database else "ERROR: Database not found."
        gr.Markdown(f"**Status:** `{status}` | **Indexed Items:** `{len(database) if database else 0}`")
    
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=2):
                query_text_input = gr.Textbox(placeholder="Enter your text query here (e.g., 'red silk dress with flowers')", label="Text Query")
                k_slider = gr.Slider(minimum=1, maximum=20, step=1, label="Number of results", value=4)
            with gr.Column(scale=1):
                query_image_input = gr.Image(type="pil", label="Or Upload an Image Query")
        search_button = gr.Button("🔍 Search All Models", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 1. ColPali Results")
            colpali_gallery = gr.Gallery(label="Retrieved by ColPali", show_label=False, columns=2, height="auto")
            colpali_text = gr.Textbox(label="ColPali Info")
        with gr.Column():
            gr.Markdown("## 2. CLIP Results (OpenAI)")
            clip_gallery = gr.Gallery(label="Retrieved by CLIP", show_label=False, columns=2, height="auto")
            clip_text = gr.Textbox(label="CLIP Info")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 3. Fashion-CLIP Results")
            fashion_clip_gallery = gr.Gallery(label="Retrieved by Fashion-CLIP", show_label=False, columns=2, height="auto")
            fashion_clip_text = gr.Textbox(label="Fashion-CLIP Info")
        with gr.Column():
            gr.Markdown("## 4. SigLIP 2 Results (Google)")
            siglip_gallery = gr.Gallery(label="Retrieved by SigLIP 2", show_label=False, columns=2, height="auto")
            siglip_text = gr.Textbox(label="SigLIP 2 Info")

    search_button.click(
        fn=search_all, 
        inputs=[query_text_input, query_image_input, k_slider],
        outputs=[
            colpali_gallery, colpali_text, 
            clip_gallery, clip_text, 
            fashion_clip_gallery, fashion_clip_text, 
            siglip_gallery, siglip_text
        ]
    )

if __name__ == "__main__":
    demo.queue().launch(debug=True)