"""Predict the emotion category of a WAV audio file."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

from audio_features import extract_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a speech WAV file.")
    parser.add_argument("audio_file", help="Path to a .wav file")
    args = parser.parse_args()

    model_path = MODEL_DIR / "emotion_model.keras"
    scaler_path = MODEL_DIR / "scaler.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    for path in (model_path, scaler_path, encoder_path):
        if not path.exists():
            raise FileNotFoundError("Model files are missing. Run python src/train_model.py first.")

    feature = extract_features(args.audio_file).reshape(1, -1)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    model = keras.models.load_model(model_path)
    probabilities = model.predict(scaler.transform(feature), verbose=0)[0]
    best = int(np.argmax(probabilities))
    print(f"Predicted emotion: {encoder.inverse_transform([best])[0]}")
    print(f"Model confidence: {probabilities[best] * 100:.2f}%")
    print("Note: confidence is not a guarantee that the emotion is correct.")


if __name__ == "__main__":
    main()
