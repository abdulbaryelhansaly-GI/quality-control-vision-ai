import cv2
import numpy as np
import os
from pathlib import Path

def load_and_preprocess(image_path, size=(224, 224)):
    """
    Charge et prétraite une image pour le CNN.
    Retourne : (image normalisée, carte de contours)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Image introuvable : {image_path}")

    # Étape 1 — BGR → RGB + resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    # Étape 2 — Niveaux de gris pour détection de contours
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Étape 3 — Flou pour réduire le bruit
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Étape 4 — Détection de contours (Canny)
    edges = cv2.Canny(blur, 50, 150)

    # Étape 5 — Normalisation float32 pour le CNN
    img_normalized = img.astype('float32') / 255.0

    return img_normalized, edges


def augment_image(img):
    """
    Augmentation de données pour enrichir le dataset.
    """
    augmented = []

    for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        augmented.append(cv2.rotate(img, angle))

    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.flip(img, 0))

    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    dark   = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
    augmented.extend([bright, dark])

    return augmented


def prepare_dataset(data_dir, output_dir, size=(224, 224)):
    """
    Prépare tout le dataset : good/ et defective/
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)

    for label in ['good', 'defective']:
        src = data_dir / label
        dst = output_dir / label
        dst.mkdir(parents=True, exist_ok=True)

        for img_path in src.glob('*.png'):
            img_norm, edges = load_and_preprocess(img_path, size)
            img_uint8 = (img_norm * 255).astype(np.uint8)
            cv2.imwrite(str(dst / img_path.name), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            if label == 'defective':
                augmented = augment_image(img_uint8)
                for i, aug in enumerate(augmented):
                    out_name = f"{img_path.stem}_aug{i}.png"
                    cv2.imwrite(str(dst / out_name), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

        print(f"[OK] {label} → {len(list(dst.glob('*.png')))} images traitées")


if __name__ == "__main__":
    prepare_dataset(data_dir="data/raw", output_dir="data/processed")
