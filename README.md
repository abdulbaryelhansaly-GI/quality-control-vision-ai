# 🏭 Industrial Quality Control System — AI Vision

> Détection automatique de défauts sur pièces industrielles
> Accuracy : **99%** | ENSA Kénitra — Génie Industriel

## 🎯 Description
Système de contrôle qualité par vision artificielle capable de détecter 
les défauts sur des bouteilles industrielles en temps réel.

## 🛠️ Stack technique
- **Modèle** : MobileNetV2 (Transfer Learning)
- **Vision** : OpenCV (détection de contours, Canny)
- **Interface** : Streamlit
- **Dataset** : MVTec AD — Bottle

## 📊 Résultats
| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Defective | 98% | 98% | 98% |
| Good | 99% | 99% | 99% |
| **Overall** | **99%** | **99%** | **99%** |

## 🚀 Installation
pip install -r requirements.txt
streamlit run app.py

## 📁 Structure
quality_control_vision/
├── src/
│   ├── preprocess.py   # OpenCV pipeline
│   ├── model.py        # MobileNetV2
│   ├── train.py        # Training
│   └── inference.py    # Real-time detection
├── app.py              # Streamlit interface
└── requirements.txt

## 👨‍🎓 Auteur
**[Ton Nom]** — Étudiant Génie Industriel, ENSA Kénitra
