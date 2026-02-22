# SLFI_Task2
# Multimodal Speech Emotion Recognition using HuBERT, BiLSTM and BERT

## Project Overview
This project builds a multimodal emotion recognition system using speech and text modalities.

Dataset: Toronto Emotional Speech Set (TESS)

Models:
- Speech pipeline: HuBERT + BiLSTM
- Text pipeline: BERT classifier
- Fusion pipeline: Speech + Text embeddings

---

## Objective
Compare performance of:
1. Speech-only model
2. Text-only model
3. Multimodal fusion model

---

## Dataset
- 2800 audio samples
- 7 emotions
- Tone-based emotion expression

---

## Model Pipelines

Speech:
Audio → HuBERT → BiLSTM → Emotion

Text:
Filename → BERT → Classifier → Emotion

Fusion:
Speech embeddings + Text embeddings → Dense → Emotion

---

## Results

Speech Accuracy: 95.35%
Text Accuracy: 14.59%
Fusion Accuracy: 98.57%

---

## Observations

- Speech dominates emotion prediction
- Text carries minimal emotional information
- Fusion depends on strength of modalities

---

## Project Structure

project/
 ├── models/
 │   ├── speech_pipeline/
 │   ├── text_pipeline/
 │   └── fusion_pipeline/
 │
 ├── Results/
 │   ├── accuracy_tables.csv
 │   └── plots/
 │
 ├── README.md
 └── requirements.txt

---

## Technologies Used

Python, PyTorch, Transformers, HuBERT, BERT, Librosa, Scikit-learn

---

## Author

Ameer Sayyad
AI & ML – Data Science
