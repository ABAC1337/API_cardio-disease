from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Init Flask app
app = Flask(__name__)
CORS(app)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Cardiovascular Disease Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# Load model and dummy scaler
try:
    model = joblib.load('model_rf.pkl')
    dummy_data = pd.DataFrame(np.random.rand(1, 12), columns=[
        'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
        'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
        'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
    ])
    scaler = MinMaxScaler().fit(dummy_data)  # Replace with actual scaler in production
    print("Model and dummy scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or creating scaler: {e}")
    model = None
    scaler = None

label_names = ['Absence of Heart Disease', 'Presence of Heart Disease']
label_map = {'Absence of Heart Disease': 0, 'Presence of Heart Disease': 1}

@app.route('/')
def home():
    return "Cardiovascular Disease Prediction API"

@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'summary': 'Predict risk of heart disease',
    'description': 'Accepts clinical input data and returns prediction with probabilities.',
    'parameters': [
        {
            'name': 'features',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'features': {
                        'type': 'object',
                        'properties': {
                            'age': {'type': 'number', 'example': 52},
                            'gender': {'type': 'number', 'example': 1},
                            'chestpain': {'type': 'number', 'example': 0},
                            'restingBP': {'type': 'number', 'example': 130},
                            'serumcholestrol': {'type': 'number', 'example': 200},
                            'fastingbloodsugar': {'type': 'number', 'example': 0},
                            'restingrelectro': {'type': 'number', 'example': 1},
                            'maxheartrate': {'type': 'number', 'example': 160},
                            'exerciseangia': {'type': 'number', 'example': 0},
                            'oldpeak': {'type': 'number', 'example': 1.0},
                            'slope': {'type': 'number', 'example': 2},
                            'noofmajorvessels': {'type': 'number', 'example': 0}
                        },
                        'required': [
                            'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
                            'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
                            'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
                        ]
                    }
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'prediction_label_numeric': {'type': 'integer', 'example': 0},
                    'prediction_label_string': {'type': 'string', 'example': 'Absence of Heart Disease'},
                    'probabilities': {
                        'type': 'object',
                        'properties': {
                            'Absence of Heart Disease': {'type': 'number', 'example': 0.87},
                            'Presence of Heart Disease': {'type': 'number', 'example': 0.13}
                        }
                    }
                }
            }
        },
        400: {
            'description': 'Missing feature in input',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        },
        500: {
            'description': 'Server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    }
})
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded."}), 500

    try:
        data = request.get_json()
        input_features_dict = data['features']
        input_features_order = [
            'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
            'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
            'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
        ]

        input_values = [input_features_dict[feature] for feature in input_features_order]
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input_array = scaler.transform(input_array)


        prediction_label_str = model.predict(scaled_input_array)[0]
        prediction_label_numeric = label_map.get(prediction_label_str, -1)  
        prediction_proba = model.predict_proba(scaled_input_array)[0]
        probabilities = {
            label: float(prob)
            for label, prob in zip(model.classes_, prediction_proba)
        }
        response = {
            'prediction_label_numeric': int(prediction_label_numeric),
            'prediction_label_string': prediction_label_str,
            'probabilities': probabilities
        }

        return jsonify(response)
    except KeyError as e:
        return jsonify({"error": f"Missing feature in input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
