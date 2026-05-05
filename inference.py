import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from preprocess import load_and_preprocess

MODEL_PATH = "models/quality_control_best.h5"
LABELS     = ['defective', 'good']
THRESHOLD  = 0.85


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, image_path):
    img_norm, edges = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img_norm, axis=0)

    probs = model.predict(img_batch, verbose=0)[0]
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]
    label = LABELS[class_idx]

    return {
        "label": label,
        "confidence": float(confidence),
        "probabilities": {l: float(p) for l, p in zip(LABELS, probs)},
        "edges": edges,
        "is_defective": label == 'defective',
        "above_threshold": confidence >= THRESHOLD
    }


def annotate_image(image_path, result):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (224, 224))

    color = (0, 0, 255) if result['is_defective'] else (0, 200, 0)
    label_text = f"{result['label'].upper()} ({result['confidence']*100:.1f}%)"

    cv2.rectangle(img, (0, 0), (224, 40), color, -1)
    cv2.putText(img, label_text, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


def run_on_folder(folder_path):
    model = load_model()
    folder = Path(folder_path)
    results = []

    for img_path in folder.glob("*.png"):
        result = predict_image(model, img_path)
        result['file'] = img_path.name
        results.append(result)

        status = "DEFAUT" if result['is_defective'] else "OK"
        print(f"{img_path.name:<30} {status}  ({result['confidence']*100:.1f}%)")

    defects = sum(1 for r in results if r['is_defective'])
    print(f"\nResume : {defects}/{len(results)} pieces defectueuses")
    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/test"
    run_on_folder(path)
