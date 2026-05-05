import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import load_and_preprocess
from inference import predict_image, annotate_image, load_model

st.set_page_config(
    page_title="Contrôle Qualité IA",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Système de Contrôle Qualité par Vision")
st.markdown("**ENSA Kénitra — Génie Industriel | Projet IA**")
st.divider()

@st.cache_resource
def get_model():
    return load_model()

with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider("Seuil de confiance", 0.5, 1.0, 0.85, 0.01)
    show_edges = st.checkbox("Afficher la carte de contours", value=True)
    st.divider()
    st.info("Dataset : MVTec AD\nModèle : EfficientNetB0\nTransfer Learning + Fine-tuning")

uploaded = st.file_uploader(
    "Dépose une image de pièce industrielle",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded:
    model = get_model()

    for file in uploaded:
        col1, col2, col3 = st.columns([1, 1, 1])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        result = predict_image(model, tmp_path)

        with col1:
            st.subheader("Image originale")
            st.image(tmp_path, use_column_width=True)

        if show_edges:
            with col2:
                st.subheader("Contours (Canny)")
                st.image(result['edges'], use_column_width=True, clamp=True)

        with col3:
            st.subheader("Résultat")
            if result['is_defective']:
                st.error("❌ DÉFAUT DÉTECTÉ")
            else:
                st.success("✅ PIÈCE CONFORME")

            conf = result['confidence'] * 100
            st.metric("Confiance", f"{conf:.1f}%")
            st.progress(result['confidence'])

            st.markdown("**Probabilités :**")
            for label, prob in result['probabilities'].items():
                st.write(f"- `{label}` : {prob*100:.1f}%")

        os.unlink(tmp_path)
        st.divider()

else:
    st.info("👆 Upload une ou plusieurs images pour lancer l'analyse")
