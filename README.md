# 🧬 Diagnovista – AI-Powered Disease & Diabetes Prediction System

Diagnovista is a full-stack machine learning web application that predicts possible diseases based on user-reported symptoms and estimates diabetes risk using clinical health indicators.

It combines classical machine learning, intelligent feature engineering, and interactive visual analytics to deliver an intuitive diagnostic experience.

---

## 🚀 Features

### 🩺 1. Disease Prediction System
- Multi-label symptom encoding using `MultiLabelBinarizer`
- Random Forest classifier with class balancing
- Custom symptom importance weighting logic
- Top-3 probabilistic disease predictions
- Interactive confidence visualization
- Symptom importance bar chart
- Symptom-disease relationship network graph

### 🩸 2. Diabetes Risk Prediction
- Feature selection using Random Forest (`SelectFromModel`)
- Feature scaling using `StandardScaler`
- Risk probability gauge visualization
- Interactive feature value comparison chart

### 🌐 3. Web Interface
- Flask backend
- HTML, CSS, and JavaScript frontend
- Dynamic symptom grouping
- Interactive Plotly visualizations

---

## 🧠 Machine Learning Techniques Used

- Random Forest Classification
- Multi-Label Feature Encoding
- Feature Selection
- Feature Scaling
- Class Balancing
- Probabilistic Prediction
- Custom Feature Weighting Strategy

---

## 📊 Visual Analytics

- Disease Confidence Bar Plot
- Symptom Importance Visualization
- Symptom-Disease Network Graph
- Diabetes Risk Probability Gauge

---

## 🛠 Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas / NumPy
- Plotly
- HTML / CSS / JavaScript

---



## ⚙️ Installation & Setup

1. Clone the repository:

```
git clone https://github.com/yourusername/Diagnovista.git
cd Diagnovista
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Train models:

```
python train_diseases_model.py
python train_diabetes_model.py
```

4. Run the application:

```
python main.py
```

5. Open in browser:

```
http://127.0.0.1:5000
```

---

## 📈 Future Improvements

- Add deep learning-based symptom classification
- Integrate SHAP for model explainability
- Deploy using Docker & cloud services
- Add user authentication and history tracking
- Improve UI/UX with advanced visual design

---

## ⚠ Disclaimer

Diagnovista is a machine learning project for educational and research purposes only. It is not a substitute for professional medical advice.

---

## 👨‍💻 Author

Developed by Nilesh singh
