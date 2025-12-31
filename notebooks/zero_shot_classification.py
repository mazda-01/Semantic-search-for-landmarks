import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

INDEX_FILE = '../for-models/faiss.index'
EMBEDDING_FILE = '../for-models/embedding.pkl'
DATASET_DIR = '../dataset_showplace'
DEVICE = 'mps'

class_prompts = [
    "Ancient Egyptian pyramids in the desert",
    "A very tall modern skyscraper or historical tower",
    "Hot air balloons flying over rock formations in Cappadocia",
    "Scenic waterfalls and turquoise lakes in a national park",
    "Colorful layered rock formations and rainbow mountains",
    "Ancient ruins of a lost civilization in the mountains",
    "Red brick walls and golden domes of the Moscow Kremlin",
    "Tall pillar-like mountains with mist and clouds, like in Avatar movie"
]

class_names = [
    "Ancient pyramids",
    "Tall towers/skyscrapers",
    "Cappadocia balloons",
    "Waterfalls and lakes",
    "Colorful mountains",
    "Ancient ruins",
    "Moscow Kremlin",
    "Pillar mountains"
]

model_name = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.to(DEVICE)
model.eval()


with open(EMBEDDING_FILE, 'rb') as f:
    image_paths = pickle.load(f)


text_inputs = processor(text=class_names, padding=True, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)


predictions = []
scores = []

for idx, path in enumerate(image_paths):
    try:
        image = Image.open(path).convert('RGB')
        image_inputs = processor(images=image, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0)
        max_sc, max_idx = similarities.max(dim=0)

        predicted_class = class_names[max_idx]
        predictions.append(predicted_class)
        scores.append(max_sc.item())

    except Exception as e:
        print(f"Ошибка с {path}: {e}")
        predictions.append("Unknown")
        scores.append(0.0)

counter = Counter(predictions)
print("\nРаспределение по классам:")
for class_name, count in counter.most_common():
    print(f"{class_name}: {count} изображений ({count/len(predictions)*100:.1f}%)")

plt.figure(figsize=(12, 6))
classes, counts = zip(*counter.most_common())
y_pos = np.arange(len(classes))
plt.barh(y_pos, counts, align='center')
plt.yticks(y_pos, classes)
plt.xlabel('Количество изображений')
plt.title('Zero-shot классификация датасета')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

import pandas as pd
results_df = pd.DataFrame({
    'filename': [os.path.basename(p) for p in image_paths],
    'predicted_class': predictions,
    'confidence': scores
})
results_df.to_csv('zero_shot_classification_results.csv', index=False)
print("Результаты сохранены в zero_shot_classification_results.csv")