import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('datasets/your_training_data.csv')  # <-- replace with correct file name!

X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=20)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained! Test Accuracy: {accuracy * 100:.2f}%")

os.makedirs('models', exist_ok=True)

pickle.dump(svc, open('models/svc.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))
symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}
pickle.dump(symptoms_dict, open('models/symptoms_dict.pkl', 'wb'))

print("✅ Model, Label Encoder, and Symptoms Dictionary saved successfully!")
