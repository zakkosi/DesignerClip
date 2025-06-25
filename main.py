import os
import spaces
import base64
from io import BytesIO

import gradio as gr
import torch

from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor


@spaces.GPU
def install_fa2():
    print("Install FA2")
    os.system("pip install flash-attn --no-build-isolation")
# install_fa2()


model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
        # attn_implementation="flash_attention_2", # should work on A100
    ).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")


def encode_image_to_base64(image):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

def query_gpt4o_mini(query, images, api_key):
    """Calls OpenAI's GPT-4o-mini with the query and image data."""

    if api_key and api_key.startswith("sk"):
        try:
            from openai import OpenAI
        
            base64_images = [encode_image_to_base64(image[0]) for image in images]
            client = OpenAI(api_key=api_key.strip())
            PROMPT = """
            You are a smart assistant designed to answer questions about a PDF document.
            You are given relevant information in the form of PDF pages. Use them to construct a short response to the question, and cite your sources (page numbers, etc).
            If it is not possible to answer using the provided pages, do not attempt to provide an answer and simply say the answer is not present within the documents.
            Give detailed and extensive answers, only containing info in the pages you are given.
            You can answer using information contained in plots and figures if necessary.
            Answer in the same language as the query.
            
            Query: {query}
            PDF pages:
            """
        
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(query=query)
                        }] + [{
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{im}"
                                },
                        } for im in base64_images]
                }
                ],
            max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return "OpenAI API connection failure. Verify the provided key is correct (sk-***)."
        
    return "Enter your OpenAI API key to get a custom response"


@spaces.GPU
def search(query_text: str, query_image, ds, images, k, api_key):
    if not ds:
        return [], "Please index some documents first.", [], []
    
    k = min(k, len(ds))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        
    qs = []
    llm_query = ""

    with torch.no_grad():
        if query_image is not None:
            print("Performing image query...")
            batch_query = processor.process_images([query_image]).to(model.device)
            embeddings_query = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
            llm_query = "Find documents visually similar to the uploaded image."
        
        elif query_text and query_text.strip():
            print("Performing text query...")
            batch_query = processor.process_queries([query_text]).to(model.device)
            embeddings_query = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
            llm_query = query_text
        
        else:
            return [], "Please provide a text query or an image query to search.", [], []

    scores = processor.score(qs, ds, device=device)

    top_k_indices = scores[0].topk(k).indices.tolist()

    results = []
    for idx in top_k_indices:
        results.append((images[idx], f"Page {idx}"))

    # Generate response from GPT-4o-mini
    ai_response = query_gpt4o_mini(llm_query, results, api_key)

    return results, ai_response

def index(files, ds):
    print("Converting files")
    images = convert_image_files(files)
    print(f"Files converted with {len(images)} images.")
    return index_gpu(images, ds)
    
def convert_image_files(files):
    images = []
    for f in files:
        image = Image.open(f.name).convert("RGB")
        images.append(image)

    #if len(images) >= 150:
    #    raise gr.Error("The number of images in the dataset should be less than 150.")
    return images    

def convert_pdf_files(files):
    images = []
    for f in files:
        images.extend(convert_from_path(f, thread_count=4))

    #if len(images) >= 150:
    #    raise gr.Error("The number of images in the dataset should be less than 150.")
    return images


@spaces.GPU
def index_gpu(images, ds):
    """Example script to run inference with ColPali (ColQwen2)"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        
    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        # num_workers=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x).to(model.device),
    )
    
    ds.clear()

    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return f"Uploaded and converted {len(images)} pages", ds, images



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval with Vision Language Models (ColQwen2) 📚")
    gr.Markdown("""Demo to test ColQwen2 (ColPali) on PDF documents. 
    ColPali is model implemented from the [ColPali paper](https://arxiv.org/abs/2407.01449).

    This demo allows you to upload PDF files and search for the most relevant pages based on your text or image query.
    Refresh the page if you change documents !

    ⚠️ This demo uses a model trained exclusively on A4 PDFs in portrait mode, containing english text. Performance is expected to drop for other page formats and languages.
    Other models will be released with better robustness towards different languages and document formats !
    """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 1️⃣ Upload Documents")
            #file = gr.File(file_types=["pdf"], file_count="multiple", label="Upload PDFs")
            file = gr.File(file_types=["image"], file_count="multiple", label="Upload Images")
            #file = gr.File()

            convert_button = gr.Button("🔄 Index documents")
            message = gr.Textbox("Files not yet uploaded", label="Status")
            api_key = gr.Textbox(placeholder="Enter your OpenAI KEY here (optional)", label="API key")
            embeds = gr.State(value=[])
            imgs = gr.State(value=[])

        with gr.Column(scale=3):
            gr.Markdown("## 2️⃣ Search")

            gr.Markdown("#### Search with an Image")
            query_image_input = gr.Image(type="pil", label="Upload an Image for Query")
            
            gr.Markdown("#### Or Search with Text")
            query_text_input = gr.Textbox(placeholder="Enter your text query here", label="Text Query")
            k = gr.Slider(minimum=1, maximum=10, step=1, label="Number of results", value=5)


    # Define the actions
    search_button = gr.Button("🔍 Search", variant="primary")
    output_gallery = gr.Gallery(label="Retrieved Documents", height=600, show_label=True)
    output_text = gr.Textbox(label="AI Response", placeholder="Generated response based on retrieved documents")

    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])
    
    search_button.click(
        search, 
        inputs=[query_text_input, query_image_input, embeds, imgs, k, api_key],
        outputs=[output_gallery, output_text]
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True)