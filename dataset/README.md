# Handwritten Character Recognition using CNN

## Objective
Recognize handwritten digits (0–9) from images using a Convolutional Neural Network (CNN).

## Included files
- `handwritten_character_recognition.ipynb`: notebook for data preparation, CNN training, evaluation, graphs, and prediction.
- `dataset/digits_demo.csv`: compact CSV copy of the demo images (label + 64 pixel columns).
- `requirements.txt`: Python dependencies.
- `Project_Report.md`: short report.

## Dataset
The notebook uses scikit-learn's built-in handwritten digits dataset (1,797 8×8 grayscale digit images), which is loaded directly by the notebook. An optional MNIST 28×28 setup is documented in the notebook.

## Run in VS Code
1. Extract this project folder.
2. Open the folder in VS Code.
3. In the terminal run `pip install -r requirements.txt`.
4. Open `handwritten_character_recognition.ipynb`.
5. Select a Python kernel and run cells from top to bottom.

## Scope
This starter recognizes digits 0–9, not all alphabet characters or whole words. EMNIST can be used for letters; CRNN and a word-level dataset are needed for full-word recognition.

## GitHub
Upload the extracted files and `dataset` folder, not only the ZIP.
