"""Train and evaluate a neural network on RAVDESS speech audio."""
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

from audio_features import extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# RAVDESS filename emotion field: 01-01-EMOTION-...
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}


def emotion_from_filename(filename):
    parts = Path(filename).stem.split("-")
    if len(parts) < 3:
        return None
    return EMOTIONS.get(parts[2])


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError("Create data/ and place the extracted RAVDESS Actor_* folders inside it.")
    wav_files = sorted(DATA_DIR.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError("No .wav files found. Check the dataset instructions in README.md.")

    features, labels = [], []
    print(f"Found {len(wav_files)} WAV files. Extracting MFCC features...")
    for i, path in enumerate(wav_files, start=1):
        label = emotion_from_filename(path.name)
        if label is None:
            continue
        try:
            features.append(extract_features(path))
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
        if i % 100 == 0:
            print(f"Processed {i}/{len(wav_files)} files")

    if len(features) < 10:
        raise RuntimeError("Not enough usable audio files were found.")
    X = np.asarray(features, dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Stratification keeps each emotion represented in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(len(encoder.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(
        X_train, y_train, validation_split=0.15, epochs=40, batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "emotion_model.keras")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title("RAVDESS Speech Emotion Recognition")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    print("\nSaved model files in models/ and confusion matrix in outputs/.")


if __name__ == "__main__":
    main()
