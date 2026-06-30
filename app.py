import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
import numpy as np

IMG_SIZE = (160, 160)
CLASS_NAMES = ["cat", "dog"]   # cats=0, dogs=1

@st.cache_resource
def get_model():
    return load_model("cats_dogs_model.keras")

st.title(" Classificateur Chats vs Chiens")
st.write("Charge une image de chat ou de chien.")

uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image chargée", use_container_width=True)

    # IMPORTANT : MobileNetV2 attend un prétraitement dans [-1, 1]
    # (preprocess_input), PAS un simple /255. C'est ce qui garantit
    # des prédictions correctes en ligne.
    img = ImageOps.fit(image, IMG_SIZE)
    arr = preprocess_input(np.array(img, dtype="float32"))
    arr = np.expand_dims(arr, axis=0)

    proba = float(get_model().predict(arr)[0][0])
    label = CLASS_NAMES[int(proba > 0.5)]
    conf = proba if proba > 0.5 else 1 - proba
    st.subheader(f"Prédiction : **{label}**  ({conf:.1%} de confiance)")
