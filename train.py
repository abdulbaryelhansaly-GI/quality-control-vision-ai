import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import build_model, unfreeze_top_layers

DATA_DIR    = "data/processed"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS_P1   = 20
EPOCHS_P2   = 10
MODEL_PATH  = "models/quality_control_best.keras"


def get_generators():
    gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        shear_range=0.2,
        validation_split=0.2
    )

    val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = gen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )

    n_defective = len(os.listdir(os.path.join(DATA_DIR, 'defective')))
    n_good      = len(os.listdir(os.path.join(DATA_DIR, 'good')))
    total       = n_defective + n_good

    class_indices = train_data.class_indices
    print("Classes detectees:", class_indices)

    class_weight = {}
    for label, idx in class_indices.items():
        if label == 'defective':
            class_weight[idx] = total / (2 * n_defective)
        else:
            class_weight[idx] = total / (2 * n_good)

    print(f"Class weights: {class_weight}")
    print(f"Images -> good: {n_good}, defective: {n_defective}")

    return train_data, val_data, class_weight


def get_callbacks():
    return [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.3, patience=4, min_lr=1e-7, verbose=1)
    ]


def train():
    train_data, val_data, class_weight = get_generators()
    model, base_model = build_model()

    print("\n=== Phase 1 : Entrainement de la tete ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history1 = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS_P1, callbacks=get_callbacks(),
        class_weight=class_weight
    )

    print("\n=== Phase 2 : Fine-tuning ===")
    unfreeze_top_layers(base_model, num_layers=20)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history2 = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS_P2, callbacks=get_callbacks(),
        class_weight=class_weight
    )

    return model, history1, history2, val_data


def evaluate(model, val_data):
    os.makedirs("outputs", exist_ok=True)
    val_data.reset()
    preds = model.predict(val_data, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_data.classes
    labels = list(val_data.class_indices.keys())

    print("\n=== Rapport de classification ===")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Matrice de confusion")
    plt.ylabel("Reel"); plt.xlabel("Predit")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.show()


def plot_history(h1, h2):
    os.makedirs("outputs", exist_ok=True)
    acc  = h1.history['accuracy'] + h2.history['accuracy']
    val  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, acc, label='Train')
    ax1.plot(epochs, val, label='Validation')
    ax1.axvline(EPOCHS_P1, color='red', linestyle='--', label='Fine-tuning start')
    ax1.set_title('Accuracy'); ax1.set_ylim([0, 1]); ax1.legend()
    ax2.plot(epochs, loss, label='Train Loss')
    ax2.set_title('Loss'); ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=150)
    plt.show()
