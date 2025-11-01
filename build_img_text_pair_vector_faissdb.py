import json
import os
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import base64
from io import BytesIO

text_model = ''
clip_model = ''
clip_processor = ''
device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = text_model.to(device)
clip_model = clip_model.to(device)


def encode_image(image_base64):
    image = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy()

def encode_text(text):
    return text_model.encode(text)

all_embeddings = []
all_docs = []

print("Processing image-caption pairs...")
with open("data/pmc_data/pmc_nohum_training_dataset.jsonl") as f:
    num_lines = sum(1 for _ in f)
    f.seek(0) 
    for line in tqdm(f, desc="Image paragraphs", total=num_lines):
        obj = json.loads(line)
        caption_vec = encode_text(obj["context_text"])
        image_vec = encode_image(obj["image_base64"])
        joint_vec = np.concatenate([caption_vec, image_vec])[:1024] 
        all_docs.append({"type": "image_caption_pair", **obj})
        all_embeddings.append(joint_vec)

index = faiss.IndexFlatL2(1024)
index.add(np.vstack(all_embeddings))
faiss.write_index(index, "retrieve/pmc_nohum_training_base.faiss")

with open("pmc_nohum_training_base.jsonl", "w") as f:
    for meta in all_docs:
        f.write(json.dumps(meta) + "\n")
