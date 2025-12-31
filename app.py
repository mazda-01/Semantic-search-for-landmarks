import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import os
import numpy as np

st.set_page_config(page_title="Семантический поиск достопримечательностей", layout="wide")

INDEX_FILE = 'for-models/faiss.index'          
EMBEDDING_FILE = 'for-models/embedding.pkl'    
DATASET_DIR = 'dataset_showplace'   

model_name = "openai/clip-vit-large-patch14"
DEVICE = 'cpu'  

@st.cache_resource
def load_model_and_index():
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    index = faiss.read_index(INDEX_FILE)
    with open(EMBEDDING_FILE, 'rb') as f:
        image_paths = pickle.load(f)
    
    return model, processor, index, image_paths

model, processor, index, image_paths = load_model_and_index()

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

@st.cache_resource
def get_text_embeddings():
    inputs = processor(text=class_prompts, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu()

text_features = get_text_embeddings()

st.title("🌍 Семантический поиск по достопримечательностям")
st.markdown("Датасет: 1000+ фото знаменитых мест и природных чудес")

tab1, tab2, tab3 = st.tabs(["🔍 Поиск по фото", "✍️ Поиск по тексту", "🏷️ Классификация"])

# ================= TAB 1: Image → Image =================
with tab1:
    st.header("Загрузи фото — найду похожие")
    uploaded_file = st.file_uploader("Выбери изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="Твоё фото", width=300)

        with st.spinner("Ищу похожие..."):
            inputs = processor(images=query_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                query_features = model.get_image_features(**inputs)
                query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
                query_emb = query_features.cpu().numpy().astype('float32')

            scores, indices = index.search(query_emb, 12)
            
            cols = st.columns(4)
            for i, col in enumerate(cols):
                idx = indices[0][i]
                score = scores[0][i]
                path = image_paths[idx]
                img = Image.open(path)
                with col:
                    st.image(img)
                    st.caption(f"{i+1}. Score: {score:.3f}\n{os.path.basename(path)}")

# ================= TAB 2: Text → Image =================
with tab2:
    st.header("Напиши описание — найду фото")
    text_query = st.text_input("Текст запроса", value="Эйфелева башня ночью")
    
    if st.button("Найти по тексту") or text_query:
        with st.spinner("Ищу по описанию..."):
            inputs = processor(text=text_query, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                text_emb = model.get_text_features(**inputs)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                text_emb = text_emb.cpu().numpy().astype('float32')

            scores, indices = index.search(text_emb, 12)

            cols = st.columns(4)
            for i, col in enumerate(cols):
                idx = indices[0][i]
                score = scores[0][i]
                path = image_paths[idx]
                img = Image.open(path)
                with col:
                    st.image(img)
                    st.caption(f"{i+1}. Score: {score:.3f}\n{os.path.basename(path)}")

# ================= TAB 3: Zero-shot классификация =================
with tab3:
    st.header("Загрузи фото — скажу, что это")
    uploaded_class = st.file_uploader("Фото для классификации", type=["jpg", "jpeg", "png"], key="class")

    if uploaded_class is not None:
        query_img = Image.open(uploaded_class).convert("RGB")
        st.image(query_img, caption="Твоё фото", width=300)

        with st.spinner("Классифицирую..."):
            inputs = processor(images=query_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                img_features = model.get_image_features(**inputs)
                img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

            similarities = (img_features @ text_features.T).squeeze(0)
            max_score, max_idx = similarities.max(dim=0)
            predicted = class_names[max_idx]
            confidence = max_score.item()

            st.success(f"**Предсказанный класс:** {predicted}")
            st.info(f"**Уверенность:** {confidence:.3f}")

            # Топ-3 классов
            top3_vals, top3_idx = similarities.topk(3)
            st.write("Топ-3 возможных классов:")
            for i in range(3):
                cls = class_names[top3_idx[i]]
                score = top3_vals[i].item()
                st.write(f"{i+1}. {cls} — {score:.3f}")