from flask import Flask, render_template, request
import numpy as np
import joblib
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import gdown
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

app = Flask(__name__)

MODEL_DIR = "models"

FILES = {
    "diagnovista_model.joblib": "https://drive.google.com/uc?id=1EgHMOs294ixjtc1IxEkBG2sy8yV2JUEX",
    "diabetes_model.joblib": "https://drive.google.com/uc?id=1KGznrLyFUXNGktAkj-ozKuxMvrIfHKdn",
    "diabetes_scaler.joblib": "https://drive.google.com/uc?id=1r0Cd8Px15SHl0OxfemOgY2NK5EthZWRK"
}

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, url in FILES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            gdown.download(url, path, quiet=False, fuzzy=True)

def load_models():
    download_models()

    disease_components = joblib.load('models/diagnovista_model.joblib')
    diabetes_model = joblib.load('models/diabetes_model.joblib')
    diabetes_scaler = joblib.load('models/diabetes_scaler.joblib')

    return {
        'disease': {
            'model': disease_components['model'],
            'mlb': disease_components['mlb'],
            'disease_labels': disease_components['disease_labels'],
            'symptom_weights': disease_components['symptom_weights']
        },
        'diabetes': {
            'model': diabetes_model,
            'scaler': diabetes_scaler,
            'feature_names': ['HighBP','BMI','GenHlth','PhysHlth','MentHlth','Age','Education','Income']
        }
    }

# Lazy loading to avoid Render memory crash
models = None

@app.route('/')
def home():
    global models
    if models is None:
        models = load_models()

    symptoms_grouped = {}

    for symptom in sorted(models['disease']['mlb'].classes_):
        first_letter = symptom[0].upper()
        if first_letter not in symptoms_grouped:
            symptoms_grouped[first_letter] = []
        symptoms_grouped[first_letter].append(symptom)

    return render_template(
        'index.html',
        symptoms_grouped=symptoms_grouped,
        diabetes_features=models['diabetes']['feature_names']
    )

@app.route('/predict/disease', methods=['POST'])
def predict_disease():

<<<<<<< HEAD
=======
    global models
    if models is None:
        models = load_models()

>>>>>>> ce2ac15 (lazy load models to fix memory crash)
    selected_symptoms = request.form.getlist('symptoms')

    mlb = models['disease']['mlb']
    input_data = np.zeros(len(mlb.classes_))

    for symptom in selected_symptoms:
        if symptom in mlb.classes_:
            idx = list(mlb.classes_).index(symptom)
            input_data[idx] = models['disease']['symptom_weights'].get(symptom,1)

    probas = models['disease']['model'].predict_proba([input_data])[0]
    top_indices = np.argsort(probas)[-3:][::-1]

    predictions = [{
        'name': models['disease']['disease_labels'].cat.categories[i],
        'confidence': float(probas[i]),
        'confidence_percent': f"{probas[i]:.1%}"
    } for i in top_indices]

    plots = {
        'confidence': create_disease_confidence_plot(predictions),
        'symptoms': create_symptom_importance_plot(selected_symptoms),
        'symptom_network': create_symptom_network_plot(selected_symptoms,predictions)
    }

    return render_template(
        'result.html',
        predictions=predictions,
        selected_symptoms=selected_symptoms,
        plots=plots,
        prediction_type='disease'
    )

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():

    global models
    if models is None:
        models = load_models()

    input_data = np.array([
        float(request.form['HighBP']),
        float(request.form['BMI']),
        float(request.form['GenHlth']),
        float(request.form['PhysHlth']),
        float(request.form['MentHlth']),
        float(request.form['Age']),
        float(request.form['Education']),
        float(request.form['Income'])
    ]).reshape(1,-1)

    scaled_data = models['diabetes']['scaler'].transform(input_data)

    model = models['diabetes']['model']

    prediction = int(model.predict(scaled_data)[0])
    probability = float(model.predict_proba(scaled_data)[0][1])

    plots = {
        'gauge': create_diabetes_gauge(probability),
        'feature_importance': create_feature_importance_plot(input_data[0])
    }

    return render_template(
        'result.html',
        prediction=prediction,
        probability=probability,
        plots=plots,
        input_data=input_data[0],
        feature_names=models['diabetes']['feature_names'],
        prediction_type='diabetes'
    )

def create_disease_confidence_plot(predictions):

    fig = px.bar(
        x=[p['confidence']*100 for p in predictions],
        y=[p['name'] for p in predictions],
        orientation='h',
        text=[p['confidence_percent'] for p in predictions],
        title='<b>Disease Prediction Confidence</b>',
        color=[p['confidence'] for p in predictions],
        color_continuous_scale='Tealrose',
        labels={'x':'Confidence (%)','y':''}
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0,100]),
        margin=dict(l=20,r=20,t=40,b=20),
        height=300
    )

    return json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

def create_symptom_importance_plot(selected_symptoms):

    symptoms=[s.replace('_',' ').title() for s in selected_symptoms]
    importance=[models['disease']['symptom_weights'].get(s,1) for s in selected_symptoms]

    fig=px.bar(
        x=importance,
        y=symptoms,
        orientation='h',
        title='<b>Symptom Importance Weights</b>',
        color=importance,
        color_continuous_scale='Purpor',
        labels={'x':'Weight','y':''}
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20,r=20,t=40,b=20),
        height=300,
        coloraxis_showscale=False
    )

    return json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

def create_symptom_network_plot(selected_symptoms,predictions):

    nodes=[]
    node_labels=[]

    for symptom in selected_symptoms:
        nodes.append(dict(
            x=np.random.uniform(0,0.5),
            y=np.random.uniform(0,1),
            size=10+models['disease']['symptom_weights'].get(symptom,1)*5,
            label=symptom.replace('_',' ').title(),
            color='#636EFA'
        ))

    for pred in predictions:
        nodes.append(dict(
            x=np.random.uniform(0.5,1),
            y=np.random.uniform(0,1),
            size=15+pred['confidence']*30,
            label=pred['name'],
            color='#EF553B'
        ))

    fig=go.Figure()

    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(size=node['size'],color=node['color']),
            text=node['label'],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        title='<b>Symptom-Disease Network</b>',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        height=400
    )

    return json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

def create_diabetes_gauge(probability):

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text':"Diabetes Risk Probability"},
        gauge={
            'axis':{'range':[None,100]},
            'steps':[
                {'range':[0,30],'color':'green'},
                {'range':[30,70],'color':'yellow'},
                {'range':[70,100],'color':'red'}
            ]
        }
    ))

<<<<<<< HEAD
    fig.update_layout(height=300)

=======
>>>>>>> ce2ac15 (lazy load models to fix memory crash)
    return json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

def create_feature_importance_plot(input_data):

    features=models['diabetes']['feature_names']
    values=input_data.tolist()

    fig=px.bar(
        x=values,
        y=features,
        orientation='h',
        title='Input Feature Values'
    )

<<<<<<< HEAD
    fig.update_layout(height=300)

    return json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
=======
    return json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
>>>>>>> ce2ac15 (lazy load models to fix memory crash)
