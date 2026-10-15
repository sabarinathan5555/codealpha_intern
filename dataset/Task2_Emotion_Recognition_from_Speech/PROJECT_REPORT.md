# Project Report: Emotion Recognition from Speech

## 1. Introduction
Human speech carries information through tone, pitch, rhythm, and vocal energy, in addition to spoken words. Speech Emotion Recognition (SER) uses audio processing and machine learning to estimate an emotion category from these acoustic patterns.

## 2. Objective
To build a model that classifies speech recordings into emotion categories such as neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

## 3. Tools and Technologies
- Python
- Librosa for audio loading and MFCC extraction
- NumPy for numerical processing
- TensorFlow/Keras for the neural network
- Scikit-learn for data splitting, scaling, and evaluation
- Matplotlib for the confusion matrix

## 4. Dataset
The project is designed for the RAVDESS Speech Audio dataset. The audio files contain emotion labels encoded in their filenames. The dataset must be downloaded separately from the official RAVDESS page and placed in the `data/` directory.

## 5. Methodology
1. Load WAV recordings and convert them to mono audio at 22,050 Hz.
2. Extract 40 Mel-Frequency Cepstral Coefficients (MFCCs).
3. Calculate the mean and standard deviation of each coefficient to create a fixed-length feature vector.
4. Encode emotion labels and split the dataset into training and testing sets.
5. Standardize the features using a scaler fitted only on the training data.
6. Train a feed-forward neural network with ReLU hidden layers, dropout, and a softmax output.
7. Evaluate the model on the held-out test set and generate a confusion matrix.
8. Save the trained model and preprocessing objects for later predictions.

## 6. Model Architecture
- Input layer: 80 features (40 MFCC means + 40 MFCC standard deviations).
- Hidden layer 1: 256 neurons with ReLU activation.
- Dropout: 30%.
- Hidden layer 2: 128 neurons with ReLU activation.
- Dropout: 20%.
- Output layer: one softmax unit per emotion category.

## 7. Evaluation
The training script reports test accuracy and precision, recall, and F1-score for each emotion. It also saves `outputs/confusion_matrix.png`. Actual scores must be filled in after running the model; they depend on the dataset and environment. Do not report invented accuracy values.

## 8. Applications
- Human-computer interaction
- Call-centre conversation analytics (with consent)
- Educational and research prototypes
- Voice interfaces and accessibility research

## 9. Limitations and Ethics
Emotion is complex and context-dependent. Accent, language, microphone quality, background noise, acting style, and individual differences can affect predictions. The output should not be treated as a diagnosis or definitive statement about someone's feelings. Use recordings only when you have permission and explain the system's limitations.

## 10. Conclusion
This project demonstrates a complete baseline pipeline for speech emotion recognition using MFCC features and a neural network. It includes training, evaluation, model saving, and prediction for a new WAV file. Future improvements could compare CNN or LSTM models, use data augmentation, and evaluate performance across speakers and recording conditions.
