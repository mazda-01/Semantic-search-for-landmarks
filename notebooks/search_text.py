from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import torch
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
#from sentence_transformers import SentenceTransformer

TEXT_QUERY = 'Moscow'
TOP_K = 12
INDEX_FILE = '../for-models/faiss.index'
EMBEDDING_FILE = '../for-models/embedding.pkl'
DATASET_DIR = '../dataset_showplace'

DEVICE = 'mps'

model_name = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.to(DEVICE)
model.eval()

# model = SentenceTransformer('laion/CLIP-ViT-L-14-laion2B-s32B-b82K').to(DEVICE)

index = faiss.read_index(INDEX_FILE)
with open(EMBEDDING_FILE, 'rb') as f:
    image_paths = pickle.load(f)

print(f'Загружено {index.ntotal}')

inputs = processor(text=TEXT_QUERY, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_emb = text_features.cpu().numpy()

# text_emb = model.encode(TEXT_QUERY, normalize_embeddings=True)
# text_emb = text_emb.reshape(1, -1).astype('float32')

scores, indices = index.search(text_emb, TOP_K)

print(f'Запрос: {TEXT_QUERY}')
print('Топ 12 результатов:')

for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    path = image_paths[idx]
    filename = os.path.basename(path)
    print(f"{i+1:2}. Score: {score:.4f} | {filename}")

cols = 4
rows = math.ceil(TOP_K / cols)
plt.figure(figsize=(16, 4 * rows))

for i in range(TOP_K):
    idx = indices[0][i]
    score = scores[0][i]
    img_path = image_paths[idx]
    
    img = Image.open(img_path)
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"{i+1}. {score:.3f}\n{os.path.basename(img_path)}", fontsize=10)
    plt.axis('off')

plt.suptitle(f"Text-to-Image поиск: \"{TEXT_QUERY}\"", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


