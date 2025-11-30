import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings("ignore")

def train_and_save_model(dataset_path="dataset.csv", output_path="models/diagnovista_model.joblib"):
    detection_data = pd.read_csv(dataset_path)

    detection_data["Symptoms"] = detection_data.iloc[:, 1:18].apply(
        lambda row: row.dropna().tolist(), axis=1)
    detection_data["Symptoms"] = detection_data["Symptoms"].apply(
        lambda x: [symptom.strip().lower().replace("_", " ") for symptom in x if symptom])

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(detection_data["Symptoms"])

    disease_labels = detection_data["Disease"].astype("category")
    y = disease_labels.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    disease_symptom_counts = detection_data.explode("Symptoms").groupby(
        "Disease")["Symptoms"].value_counts().unstack(fill_value=0)
    disease_major_symptoms = {}

    for disease in disease_symptom_counts.index:
        symptom_frequencies = disease_symptom_counts.loc[disease].sort_values(ascending=False)
        threshold = max(2, symptom_frequencies.max() * 0.7)
        major_symptoms = set(symptom_frequencies[symptom_frequencies >= threshold].index)
        disease_major_symptoms[disease] = major_symptoms

    symptom_weights = {symptom: 1 for symptom in mlb.classes_}

    for disease, major_symptoms in disease_major_symptoms.items():
        for symptom in major_symptoms:
            symptom_weights[symptom] = 6

    symptom_disease_counts = detection_data.explode("Symptoms").groupby("Symptoms")["Disease"].nunique()
    high_overlap_symptoms = symptom_disease_counts[symptom_disease_counts > 5].index

    for symptom in high_overlap_symptoms:
        symptom_weights[symptom] = 1

    model_components = {
        'model': model,
        'mlb': mlb,
        'disease_labels': disease_labels,
        'symptom_weights': symptom_weights,
        'disease_major_symptoms': disease_major_symptoms
    }

    joblib.dump(model_components, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_and_save_model()