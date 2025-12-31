import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import pickle
from tqdm import tqdm

DATASET_DIR = '../dataset_showplace'
INDEX_FILE = '../for-models/faiss.index'
EMBEDDINGS_FILE = '../for-models/embedding.pkl'

model_name = 'openai/clip-vit-large-patch14'
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = 'mps'
model.to(device)
model.eval()

image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
embeddings = []
paths = []

with torch.no_grad():
    for path in tqdm(image_paths):
        image = Image.open(path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt').to(device)
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # Нормализация (важно для cosine)
        embeddings.append(emb.cpu().numpy())
        paths.append(path)

embeddings = np.vstack(embeddings)
print(f'Эмбеддинги готовы {embeddings.shape}')

# Построение FAISS индекса (Inner Product = cosine similarity после нормализации)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner Product для cosine
index.add(embeddings)

# Сохранение
faiss.write_index(index, INDEX_FILE)
with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump(paths, f)

print("Индекс построен и сохранён!")
