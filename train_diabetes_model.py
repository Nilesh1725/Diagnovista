import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np

def train_diabetes_model():
    # Load dataset
    df = pd.read_csv("merge_diabetes_data.csv")
    
    # Preprocessing
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature selection using Random Forest
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train, y_train)
    
    # Save mask of selected features
    selected_features_mask = selector.get_support()
    selected_feature_names = X.columns[selected_features_mask]
    
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Selected {X_train_selected.shape[1]} features out of {X.shape[1]}")
    print("Selected features are:", list(selected_feature_names))
    
    # Scale only the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train model
    model = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model, scaler, and feature selector
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/diabetes_model.joblib')
    joblib.dump(scaler, 'models/diabetes_scaler.joblib')
    joblib.dump(selector, 'models/diabetes_selector.joblib')
    joblib.dump(list(selected_feature_names), 'models/diabetes_selected_features.joblib')

    print("Model, scaler, selector, and selected feature names saved successfully.")

if __name__ == "__main__":
    train_diabetes_model()
