# 🧠 Epilepsy Prediction with EEG using Machine Learning & Deep Learning

> **Guardian AI** — An automated, explainable seizure detection framework built for clinical reliability.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-f7931e?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Epilepsy is a neurological disorder affecting approximately **50 million people worldwide**, marked by recurrent seizures caused by abnormal electrical activity in the brain. Manual EEG interpretation by neurologists is time-consuming, subjective, and prone to fatigue-related errors.

This project presents **Guardian AI**, an end-to-end automated EEG seizure detection system that:
- Benchmarks **5 classical ML models** on the UCI Epileptic Seizure Recognition Dataset
- Implements **ChronoNet (IC-DRNN)** — a Convolutional + Densely Connected GRU deep learning architecture
- Proposes **AttentionChronoNet** — a novel architecture with a Self-Attention block and Additive Attention Pooling
- Provides **clinical interpretability** via SHAP (SHapley Additive Explanations) and Attention Weight visualizations

---

## 🗂️ Repository Structure

```
.
├── chrononet/          # ChronoNet & AttentionChronoNet PyTorch implementation
├── svm/                # Support Vector Machine model and training scripts
├── random_forest/      # Random Forest model and training scripts
├── edf/                # EDF file handling and MNE-Python preprocessing utilities
├── Project report/     # Full project report (PDF)
├── .devcontainer/      # Development container configuration
└── README.md
```

---

## 🏆 Results at a Glance

### Machine Learning Models

| Model | Accuracy | Sensitivity | Specificity | F1-Score |
|---|---|---|---|---|
| **Support Vector Machine** | **98.28%** | 97.10% | 98.74% | 0.970 |
| Random Forest | 97.45% | 95.83% | 98.12% | 0.954 |
| Gaussian Naive Bayes | 95.73% | 94.20% | 96.10% | 0.930 |
| K-Nearest Neighbors | 93.88% | 92.40% | 94.20% | 0.912 |
| PCA + Logistic Regression | 91.00% | 89.50% | 91.80% | 0.876 |
| Logistic Regression | 82.76% | 81.30% | 83.50% | 0.801 |

### Deep Learning — Ablation Study

| Model | Accuracy | Missed Seizures (FN) |
|---|---|---|
| SimpleChronoNet | 98.26% | 32 |
| **AttentionChronoNet** | **99.22%** | **10** |

> AttentionChronoNet reduced missed seizures by **68.75%** compared to the baseline — a direct improvement in patient safety.

---

## 🧬 Architecture

### ChronoNet (IC-DRNN)

Combines **Inception-style Conv1D layers** with **Densely Connected GRU layers** to extract features at multiple time scales while addressing the degradation problem via dense skip connections.

```
Input (1D EEG) → Conv1D (Inception) → GRU Layer 1 → GRU Layer 2 → FC → Output
```

### AttentionChronoNet (Novel)

Extends ChronoNet with two key additions:

1. **SelfAttentionBlock** — Inserted between GRU layers; allows every EEG time-point to attend to every other time-point simultaneously via scaled dot-product attention, capturing long-range dependencies.
2. **AdditiveAttentionPool** — Replaces the last-timestep readout with a learned weighted sum over all 178 time-steps, producing extractable attention heatmaps aligned with the raw EEG signal.

```
Raw EEG → Conv1D → GRU₁ → SelfAttentionBlock → GRU₂ → AdditiveAttentionPool → FC → Output
```

---

## 📊 Dataset

**UCI Epileptic Seizure Recognition Dataset**

- **500 subjects**, each with 23.6 seconds of EEG recording
- Sampling rate: **178 Hz** → 178 features per sample
- **11,500 total instances**
- Binary classification: Class 1 (Seizure = 1) vs. Classes 2–5 (Non-Seizure = 0)

| Class | Description |
|---|---|
| 1 | Active seizure activity |
| 2 | EEG from tumor area (no seizure) |
| 3 | EEG from healthy area in tumor patients |
| 4 | Healthy subjects, eyes closed |
| 5 | Healthy subjects, eyes open |

---

## 🔍 Explainability

To enable clinical adoption, the system integrates **SHAP with GradientExplainer** to attribute each prediction to specific EEG time-points:

- **Positive SHAP value** → time-point pushes prediction toward seizure
- **Negative SHAP value** → time-point pushes toward non-seizure
- **High absolute SHAP** → most discriminative moment in the signal

SHAP peaks and attention weight peaks were validated to align with **high-amplitude, high-frequency regions** corresponding to known epileptic spike-and-wave morphology.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for ChronoNet training)

### Install Dependencies

```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn mne imbalanced-learn scipy shap jupyter
```

### Run ML Models

```bash
cd svm/
python train.py

cd ../random_forest/
python train.py
```

### Run ChronoNet / AttentionChronoNet

```bash
cd chrononet/
python train.py --model attention   # AttentionChronoNet
python train.py --model simple      # SimpleChronoNet baseline
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch, TensorFlow/Keras |
| ML | Scikit-learn |
| Signal Processing | MNE-Python, SciPy |
| Explainability | SHAP (GradientExplainer) |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, VS Code |

---

## 👨‍💻 Authors

**Abishek Ragav J** (312422243005) and **Aldan Dallas D** (312422243009)

B.Tech — Artificial Intelligence and Data Science
St. Joseph's Institute of Technology, Chennai
Anna University · April 2026

**Supervisor:** Mrs. Priya S B, MTech (Ph.D), Assistant Professor

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
