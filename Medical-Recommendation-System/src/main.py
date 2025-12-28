from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model and dictionaries
svc = pickle.load(open('models/svc.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))
symptoms_dict = pickle.load(open('models/symptoms_dict.pkl', 'rb'))

# Disease list based on label encoder
diseases_list = {index: label for index, label in enumerate(le.classes_)}

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc.astype(str))

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten().tolist() if not pre.empty else []

    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist() if not med.empty else []

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist() if not die.empty else []

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = " ".join(wrkout.astype(str)) if not wrkout.empty else ""

    return desc, pre, med, die, wrkout

# Prediction logic
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    prediction_index = svc.predict([input_vector])[0]
    predicted_disease = le.inverse_transform([prediction_index])[0]
    return predicted_disease

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')
    if not symptoms or symptoms.strip().lower() == "symptoms":
        return render_template('index.html', message="Please provide valid symptoms.")

    user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]
    predicted_disease = get_predicted_value(user_symptoms)

    dis_des, pre, meds, diet, wrkout = helper(predicted_disease)

    return render_template('index.html',
                           predicted_disease=predicted_disease,
                           dis_des=dis_des,
                           my_precautions=pre,
                           medications=meds,
                           my_diet=diet,
                           workout=wrkout)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)
