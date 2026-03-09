from flask import Flask, render_template, request
import numpy as np
import joblib
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
app = Flask(__name__)
def load_models():
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
            'feature_names': ['HighBP', 'BMI', 'GenHlth', 'PhysHlth', 'MentHlth', 'Age', 'Education', 'Income']
        }
    }
models = load_models()
@app.route('/')
def home():
    symptoms_grouped = {}
    for symptom in sorted(models['disease']['mlb'].classes_):
        first_letter = symptom[0].upper()
        if first_letter not in symptoms_grouped:
            symptoms_grouped[first_letter] = []
        symptoms_grouped[first_letter].append(symptom)
    return render_template('index.html', 
                         symptoms_grouped=symptoms_grouped,
                         diabetes_features=models['diabetes']['feature_names'])
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    selected_symptoms = request.form.getlist('symptoms')

    mlb = models['disease']['mlb']
    input_data = np.zeros(len(mlb.classes_))
    for symptom in selected_symptoms:
        if symptom in mlb.classes_:
            idx = list(mlb.classes_).index(symptom)
            input_data[idx] = models['disease']['symptom_weights'].get(symptom, 1)
    
   
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
        'symptom_network': create_symptom_network_plot(selected_symptoms, predictions)
    }

    return render_template('result.html',
                         predictions=predictions,
                         selected_symptoms=selected_symptoms,
                         plots=plots,
                         prediction_type='disease')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
   
    input_data = np.array([
        float(request.form['HighBP']),
        float(request.form['BMI']),
        float(request.form['GenHlth']),
        float(request.form['PhysHlth']),
        float(request.form['MentHlth']),
        float(request.form['Age']),
        float(request.form['Education']),
        float(request.form['Income'])
    ]).reshape(1, -1)
    
   
    scaled_data = models['diabetes']['scaler'].transform(input_data)
    prediction = models['diabetes']['model'].predict(scaled_data)[0]
    probability = models['diabetes']['model'].predict_proba(scaled_data)[0][1]
    

    plots = {
        'gauge': create_diabetes_gauge(probability),
        'feature_importance': create_feature_importance_plot(input_data[0])
    }
    
    return render_template('result.html',
                         prediction=prediction,
                         probability=probability,
                         plots=plots,
                         input_data=input_data[0],
                         feature_names=models['diabetes']['feature_names'],
                         prediction_type='diabetes')


def create_disease_confidence_plot(predictions):
    fig = px.bar(
        x=[p['confidence']*100 for p in predictions],
        y=[p['name'] for p in predictions],
        orientation='h',
        text=[p['confidence_percent'] for p in predictions],
        title='<b>Disease Prediction Confidence</b>',
        color=[p['confidence'] for p in predictions],
        color_continuous_scale='Tealrose',
        labels={'x': 'Confidence (%)', 'y': ''}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_symptom_importance_plot(selected_symptoms):
    symptoms = [s.replace('_', ' ').title() for s in selected_symptoms]
    importance = [models['disease']['symptom_weights'].get(s, 1) 
                 for s in selected_symptoms]
    
    fig = px.bar(
        x=importance,
        y=symptoms,
        orientation='h',
        title='<b>Symptom Importance Weights</b>',
        color=importance,
        color_continuous_scale='Purpor',
        labels={'x': 'Weight', 'y': ''}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        coloraxis_showscale=False
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_symptom_network_plot(selected_symptoms, predictions):
    
    nodes = []
    node_labels = []
    
   
    for i, symptom in enumerate(selected_symptoms):
        nodes.append(dict(
            x=np.random.uniform(0, 0.5),
            y=np.random.uniform(0, 1),
            size=10 + models['disease']['symptom_weights'].get(symptom, 1) * 5,
            label=symptom.replace('_', ' ').title(),
            color='#636EFA'
        ))
        node_labels.append(symptom.replace('_', ' ').title())
    
    
    for i, pred in enumerate(predictions):
        nodes.append(dict(
            x=np.random.uniform(0.5, 1),
            y=np.random.uniform(0, 1),
            size=15 + pred['confidence'] * 30,
            label=pred['name'],
            color='#EF553B'
        ))
        node_labels.append(pred['name'])
    
   
    edges = []
    for i, symptom in enumerate(selected_symptoms):
        for j, pred in enumerate(predictions):
            edges.append(dict(
                x0=nodes[i]['x'],
                y0=nodes[i]['y'],
                x1=nodes[len(selected_symptoms)+j]['x'],
                y1=nodes[len(selected_symptoms)+j]['y'],
                width=2
            ))
    
    
    fig = go.Figure()
    
   
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge['x0'], edge['x1'], None],
            y=[edge['y0'], edge['y1'], None],
            mode='lines',
            line=dict(width=edge['width'], color='#AAAAAA'),
            hoverinfo='none',
            showlegend=False
        ))
    
   
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(
                size=node['size'],
                color=node['color'],
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=node['label'],
            textposition="top center",
            hoverinfo='text',
            showlegend=False
        ))
    
    fig.update_layout(
        title='<b>Symptom-Disease Network</b>',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_diabetes_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Diabetes Risk Probability</b>", 'font': {'size': 18}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability*100}
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_feature_importance_plot(input_data):
    features = models['diabetes']['feature_names']
    values = input_data.tolist()
    
    fig = px.bar(
        x=values,
        y=features,
        orientation='h',
        title='<b>Input Feature Values</b>',
        color=values,
        color_continuous_scale='Blugrn',
        labels={'x': 'Value', 'y': ''}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        coloraxis_showscale=False
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)