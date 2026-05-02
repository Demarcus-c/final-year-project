# final-year-project
🧠 Epilepsy Detection using EEG (ML + Deep Learning)

An end-to-end intelligent system for automated epilepsy seizure detection using EEG signals, combining traditional machine learning and advanced deep learning architectures.

📌 Overview

Epilepsy is a neurological disorder affecting millions worldwide, where early and accurate detection of seizures is critical.

Manual EEG analysis is:

Time-consuming
Subjective
Error-prone

This project introduces an automated, explainable, and high-accuracy seizure detection system using both:

Machine Learning models
Deep Learning (ChronoNet + Attention mechanisms)
🚀 Key Features
✅ End-to-end EEG processing pipeline
✅ Comparison of ML vs Deep Learning models
✅ Custom AttentionChronoNet architecture
✅ High accuracy: 99.22% (best model)
✅ Reduced false negatives (critical for clinical use)
✅ Explainability using SHAP + Attention Maps
✅ Scalable for real-time and wearable applications
🏗️ System Architecture

Pipeline:

EEG Data → Preprocessing → Feature Engineering / Raw Input
        → ML Models (SVM, RF, KNN, NB)
        → Deep Learning (ChronoNet, AttentionChronoNet)
        → Evaluation + Explainability (SHAP)
📊 Dataset
UCI Epileptic Seizure Recognition Dataset
11,500 samples
178 features (EEG time points)
5 classes → converted to binary:
Seizure → 1
Non-Seizure → 0
⚙️ Tech Stack
🧪 Machine Learning
Support Vector Machine (SVM)
Random Forest
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Logistic Regression
🤖 Deep Learning
PyTorch
ChronoNet (CNN + GRU)
AttentionChronoNet (custom enhancement)
📚 Libraries
NumPy, Pandas
Scikit-learn
Matplotlib, Seaborn
SHAP
MNE-Python
🧠 Models & Performance
📌 Machine Learning Results
Model	Accuracy
SVM	98.28%
Random Forest	97.45%
Naive Bayes	95.73%
KNN	93.88%
Logistic Regression	82.76%
📌 Deep Learning Results
Model	Accuracy	Notes
ChronoNet	96.7%	Better temporal learning
AttentionChronoNet	99.22%	Best performance
🔥 Key Improvement
False negatives reduced from 32 → 10
Huge impact for real-world medical safety
🧩 Novel Contribution: AttentionChronoNet

Enhancements over standard ChronoNet:

🔹 Self-Attention Block
Captures long-range dependencies in EEG signals
🔹 Additive Attention Pooling
Uses full signal instead of last timestep
Generates interpretable importance weights
🔍 Explainability (Very Important)

To make the model clinically usable:

SHAP (Shapley Values) used for interpretation
Shows which EEG time points influenced predictions
Attention maps align with real seizure patterns
🛠️ Installation
git clone https://github.com/your-username/epilepsy-detection-eeg.git
cd epilepsy-detection-eeg
pip install -r requirements.txt
▶️ Usage
python train_ml.py
python train_dl.py
python evaluate.py

💡 Applications
🏥 Clinical decision support
⌚ Wearable seizure detection devices
🧠 Neurological disorder research
⚡ Real-time monitoring systems
⚠️ Limitations
Tested only on UCI dataset
Not yet real-time
Simplified ChronoNet version
🔮 Future Work
Real-time EEG streaming
Integration with wearable devices
Multi-class classification
Testing on larger clinical datasets (e.g., TUH EEG)
More architectures (ResNet, Transformers)
