from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import torch
import matplotlib.pyplot as plt
import math

QUERY_IMAGE = 'query.jpg'  # Твоё query-изображение (положи в корень)
INDEX_FILE = '../for-models/faiss.index'
EMBEDDINGS_FILE = '../for-models/embedding.pkl'
TOP_K = 10

# Загрузка модели
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
device = 'mps'
model.to(device)

# Загрузка индекса и путей
index = faiss.read_index(INDEX_FILE)
with open(EMBEDDINGS_FILE, 'rb') as f:
    paths = pickle.load(f)

# Эмбеддинг query
query_img = Image.open(QUERY_IMAGE).convert("RGB")
inputs = processor(images=query_img, return_tensors="pt").to(device)
with torch.no_grad():
    query_emb = model.get_image_features(**inputs)
    query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy()

# Поиск
scores, indices = index.search(query_emb, TOP_K + 1)  # +1 чтобы исключить самого себя если query в датасете
print("Top-K похожих:")
for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    if i == 0 and score > 0.99:  # Пропустить если это само query
        continue
    print(f"{i+1}. Score: {score:.4f} | {paths[idx]}")

# Визуализация результатов
total_images = TOP_K + 1  # +1 для query
cols = 5
rows = math.ceil(total_images / cols)

plt.figure(figsize=(20, 4 * rows))  # Шире и выше

# Query изображение
plt.subplot(rows, cols, 1)
plt.imshow(query_img)
plt.title("Query Image", fontsize=14)
plt.axis('off')

# Результаты
for i in range(TOP_K):
    # Если первый результат — это само query (score ~1.0), пропускаем его
    idx = indices[0][i]
    score = scores[0][i]
    
    # Пропуск, если это точное совпадение (score очень близок к 1.0)
    if score > 0.99 and i == 0:
        print("Пропущен точный дубликат query в датасете")
        continue
    
    img_path = paths[idx]
    img = Image.open(img_path)
    
    pos = i + 2  # +1 за query, +1 потому что subplot начинается с 1
    plt.subplot(rows, cols, pos)
    plt.imshow(img)
    plt.title(f"{i+1}. Score: {score:.3f}", fontsize=12)
    plt.axis('off')

# Если остались пустые места — ничего страшного
plt.suptitle(f"Query + Top-{TOP_K} похожих изображений", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()