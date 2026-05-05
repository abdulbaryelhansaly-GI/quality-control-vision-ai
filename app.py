import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os, sys, tempfile

# Fix — fichiers a la racine
sys.path.append(os.path.dirname(__file__))
from preprocess import load_and_preprocess

st.set_page_config(page_title="Contrôle Qualité IA", page_icon="🏭", layout="wide")

st.title("🏭 Système de Contrôle Qualité par Vision")
st.markdown("**ENSA Kénitra — Génie Industriel | Projet IA**")
st.markdown("**Modèle : MobileNetV2 | Accuracy : 99% | Dataset : MVTec AD**")
st.divider()

MODEL_PATH = "models/best_mobile.keras"
LABELS     = ['defective', 'good']

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        st.success("Modèle chargé avec succès ✅")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error("❌ Modèle introuvable — uploade best_mobile.keras dans models/")
        return None

def predict(model, image_path):
    img_norm, edges = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img_norm, axis=0)
    probs     = model.predict(img_batch, verbose=0)[0]
    class_idx = np.argmax(probs)
    return {
        "label":        LABELS[class_idx],
        "confidence":   float(probs[class_idx]),
        "probs":        {l: float(p) for l, p in zip(LABELS, probs)},
        "edges":        edges,
        "is_defective": LABELS[class_idx] == 'defective'
    }

with st.sidebar:
    st.header("⚙️ Infos projet")
    st.metric("Accuracy", "99%")
    st.metric("Precision défauts", "98%")
    st.metric("Recall défauts", "98%")
    st.divider()
    st.markdown("**Stack technique**")
    st.markdown("- MobileNetV2 (Transfer Learning)")
    st.markdown("- OpenCV (Canny Edge Detection)")
    st.markdown("- TensorFlow / Keras")
    st.markdown("- Streamlit")
    st.divider()
    st.markdown("**Auteur**")
    st.markdown("Étudiant GI — ENSA Kénitra")

model = load_model()

if model:
    uploaded = st.file_uploader(
        "Dépose une image de pièce industrielle",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded:
        for file in uploaded:
            st.subheader(f"📸 {file.name}")
            col1, col2, col3 = st.columns([1, 1, 1])

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            result = predict(model, tmp_path)

            with col1:
                st.markdown("**Image originale**")
                st.image(tmp_path, use_column_width=True)

            with col2:
                st.markdown("**Contours Canny**")
                st.image(result['edges'], use_column_width=True, clamp=True)

            with col3:
                st.markdown("**Résultat**")
                if result['is_defective']:
                    st.error("❌ DÉFAUT DÉTECTÉ")
                else:
                    st.success("✅ PIÈCE CONFORME")

                st.metric("Confiance", f"{result['confidence']*100:.1f}%")
                st.progress(result['confidence'])

                st.markdown("**Probabilités**")
                for label, prob in result['probs'].items():
                    st.write(f"- `{label}` : {prob*100:.1f}%")

            os.unlink(tmp_path)
            st.divider()
    else:
        st.info("👆 Upload une image de bouteille pour tester le système")
        st.markdown("### 💡 Comment tester")
        st.markdown("""
        1. Prends une image depuis `bottle/test/broken_large/` → défaut détecté ❌
        2. Prends une image depuis `bottle/test/good/` → pièce conforme ✅
        """)
