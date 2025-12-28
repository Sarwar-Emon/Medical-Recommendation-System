# Medical Recommendation System using Machine Learning

## 📌 Project Overview
Healthcare decision support systems can assist in early disease identification and basic medical guidance.
This project implements a **machine learning–based medical recommendation system** that predicts possible diseases based on user-reported symptoms and recommends appropriate medicines accordingly.

The system is designed as a **decision-support tool**, not a replacement for professional medical advice.

---

## 🎯 Objectives
- Predict potential diseases based on reported symptoms
- Recommend suitable medicines for identified conditions
- Apply machine learning techniques to healthcare data
- Build a simple and user-friendly prediction application

---

## 📂 Dataset
- Type: Medical symptom–disease dataset
- Features: Patient symptoms
- Target:
  - Disease prediction
  - Associated medicine recommendation

### Preprocessing includes:
- Handling missing symptom values
- Encoding categorical features
- Feature selection for efficient prediction

---

## ⚙️ Machine Learning Approach
The system follows this pipeline:

1. User inputs symptoms
2. Symptoms are preprocessed and encoded
3. A trained classification model predicts the most likely disease
4. Recommended medicines are mapped based on prediction results

---

## 📊 Model Evaluation
The model was evaluated using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score

The trained model demonstrated reliable performance for symptom-based disease prediction on the test dataset.

---

## 💊 Example Recommendation
```text
Input Symptoms:
- Fever
- Headache
- Body Pain

Predicted Disease:
- Dengue

Recommended Medicines:
- Paracetamol
- Adequate fluid intake
- Rest

🚀 Application

The project includes a Python-based application (app.py) that allows users to:

Enter symptoms

Receive predicted disease results

View recommended medicines

🗂 Project Structure
Medical-Recommendation-System/
│── data/              # Medical datasets
│── models/            # Trained ML models (.pkl)
│── notebooks/         # EDA and experiments
│── src/               # ML pipeline and prediction logic
│── templates/         # HTML templates (if Flask app)
│── static/            # CSS / JS files (if applicable)
│── app.py             # Main application entry point
│── README.md
│── requirements.txt

🛠 Installation & Usage
pip install -r requirements.txt
python app.py

⚠️ Disclaimer

This system is intended for educational and research purposes only.
It should not be used as a substitute for professional medical diagnosis or treatment.
