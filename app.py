import os
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import keras
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE   = (160, 160)
CLASS_NAMES = ["cat", "dog"]
EMOJIS     = {"cat": "🐱", "dog": "🐶"}
COLORS     = {"cat": "#7F77DD", "dog": "#1D9E75"}

st.set_page_config(
    page_title="Cats vs Dogs",
    page_icon="🐾",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header-box {
        background: linear-gradient(135deg, #7F77DD 0%, #1D9E75 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-box h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-box p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }
    .result-box {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        border: 2px solid #e9ecef;
    }
    .result-label {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    .result-prediction {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }
    .confidence-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    .stProgress > div > div > div {
        border-radius: 8px;
    }
    .upload-hint {
        text-align: center;
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>🐾 Cats vs Dogs Classifier</h1>
    <p>Modèle MobileNetV2 — accuracy 98.2 %</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return keras.models.load_model("cats_dogs_model.keras")

def preprocess(image):
    img = ImageOps.fit(image.convert("RGB"), IMG_SIZE)
    arr = np.array(img, dtype="float32")
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

st.markdown('<p class="upload-hint">Charge une image de chat ou de chien</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True, caption="")

    with st.spinner("Analyse en cours..."):
        model  = get_model()
        arr    = preprocess(image)
        proba  = float(model.predict(arr, verbose=0)[0][0])
        label  = CLASS_NAMES[int(proba > 0.5)]
        conf   = proba if proba > 0.5 else 1 - proba
        emoji  = EMOJIS[label]
        color  = COLORS[label]
        conf_pct = int(conf * 100)
        label_fr = "Chat" if label == "cat" else "Chien"

    st.markdown(f"""
    <div class="result-box">
        <p class="result-label">Prédiction</p>
        <p class="result-prediction" style="color:{color};">{emoji} {label_fr}</p>
        <p class="confidence-label">Confiance : <strong>{conf_pct} %</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(conf)

    if conf >= 0.95:
        st.success("Le modèle est très confiant dans cette prédiction.")
    elif conf >= 0.75:
        st.info("Le modèle est assez confiant.")
    else:
        st.warning("Le modèle hésite — l'image est peut-être ambiguë.")
