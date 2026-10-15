# CodeAlpha Task 2: Emotion Recognition from Speech

A beginner-friendly speech emotion recognition project. It extracts MFCC features from audio files and trains a neural network to classify the eight emotion categories in the RAVDESS speech dataset.

## Features
- Reads `.wav` audio files from RAVDESS actor folders.
- Extracts 40 MFCC coefficients and summarizes each coefficient using mean and standard deviation.
- Trains a TensorFlow neural network.
- Reports accuracy and a classification report.
- Saves the trained model, scaler, label encoder, and a confusion-matrix plot.
- Predicts an emotion from a new `.wav` file.

## 1. Requirements
Use Python 3.10 or 3.11 if possible.

```bash
pip install -r requirements.txt
```

If `librosa` installation fails on Windows, upgrade pip first:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get the dataset
Use the RAVDESS Speech Audio dataset from its official project page:
https://zenodo.org/records/1188976

Download and extract the **Audio_Speech_Actors_01-24.zip** file. Put the extracted `Actor_01`, `Actor_02`, ... folders inside this project's `data/` folder, so the structure looks like:

```text
Task2_Emotion_Recognition_from_Speech/
├── data/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...
├── src/
├── requirements.txt
└── README.md
```

The ZIP does not include the dataset itself. You must download it separately.

## 3. Train the model
From this project folder, run:

```bash
python src/train_model.py
```

Training creates:
- `models/emotion_model.keras`
- `models/scaler.joblib`
- `models/label_encoder.joblib`
- `outputs/confusion_matrix.png`

## 4. Predict an audio file
After training, run:

```bash
python src/predict_audio.py "data/Actor_01/03-01-03-01-01-01-01.wav"
```

Replace the example path with a real `.wav` file from your downloaded dataset.

## 5. Notes
- This is an educational baseline, not a production-ready emotion detector.
- Accuracy depends on the dataset split, recording quality, and training settings.
- Speech emotion recognition can be uncertain; a model's label is not a reliable assessment of a person's actual emotional state.
- Do not upload copyrighted or private audio without permission.

## Project summary
**Objective:** Recognize emotions such as happy, sad, angry, calm, fearful, disgust, surprised, and neutral from speech audio.

**Approach:** Audio preprocessing, MFCC feature extraction, feature scaling, and a TensorFlow feed-forward neural network.

**Evaluation:** Test accuracy, classification report, and confusion matrix.
