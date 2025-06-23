from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import joblib
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Swagger configuration
app.config['SWAGGER'] = {
    'title': 'Cardiovascular Disease Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# Load model and scaler
model = joblib.load('RF_Model.pkl')
scaler = joblib.load('scaler.pkl')

# Label classes for prediction output
label_classes = ['Not Cardio', 'Cardio']


@app.route('/')
def home():
    return "Welcome to Cardiovascular Disease Prediction API. Visit /apidocs for Swagger UI."


@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'summary': 'Predict risk of cardiovascular disease',
    'description': 'Provide clinical input features to predict the risk of cardiovascular disease (Cardio or Not Cardio).',
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
                            'chestpain': {'type': 'number', 'example': 0},
                            'restingBP': {'type': 'number', 'example': 130},
                            'serumcholestrol': {'type': 'number', 'example': 200},
                            'fastingbloodsugar': {'type': 'number', 'example': 0},
                            'restingrelectro': {'type': 'number', 'example': 1},
                            'maxheartrate': {'type': 'number', 'example': 160},
                            'slope': {'type': 'number', 'example': 2},
                            'noofmajorvessels': {'type': 'number', 'example': 0}
                        },
                        'required': [
                            'chestpain',
                            'restingBP',
                            'serumcholestrol',
                            'fastingbloodsugar',
                            'restingrelectro',
                            'maxheartrate',
                            'slope',
                            'noofmajorvessels'
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
                    'prediction_label_numeric': {'type': 'integer', 'example': 1},
                    'prediction_label_string': {'type': 'string', 'example': 'Cardio'},
                    'probabilities': {
                        'type': 'object',
                        'properties': {
                            'Cardio': {'type': 'number', 'example': 0.85},
                            'Not Cardio': {'type': 'number', 'example': 0.15}
                        }
                    },
                    'timestamp': {'type': 'string', 'example': '2025-06-15T12:00:00'}
                }
            }
        },
        400: {
            'description': 'Missing or invalid input',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string', 'example': 'Missing feature in input: chestpain'}
                }
            }
        },
        500: {
            'description': 'Server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string', 'example': 'Model or scaler not loaded.'}
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
        input_features = data.get('features', {})

        required_fields = [
            'chestpain', 'restingBP', 'serumcholestrol',
            'fastingbloodsugar', 'restingrelectro',
            'maxheartrate', 'slope', 'noofmajorvessels'
        ]

        for field in required_fields:
            if field not in input_features:
                return jsonify({"error": f"Missing feature in input: '{field}'"}), 400

        values = np.array([[
            float(input_features['chestpain']),
            float(input_features['restingBP']),
            float(input_features['serumcholestrol']),
            float(input_features['fastingbloodsugar']),
            float(input_features['restingrelectro']),
            float(input_features['maxheartrate']),
            float(input_features['slope']),
            float(input_features['noofmajorvessels']),
        ]])

        values_scaled = scaler.transform(values)
        prediction = model.predict(values_scaled)[0]
        probabilities = model.predict_proba(values_scaled)[0]

        prob_dict = {
            label_classes[i]: float(probabilities[i]) for i in range(len(label_classes))
        }

        response = {
            "prediction_label_numeric": int(prediction),
            "prediction_label_string": label_classes[prediction],
            "probabilities": prob_dict,
            "timestamp": datetime.now().isoformat()
        }

        logging.info(f"Prediction request: {input_features} => {response}")
        return jsonify(response)

    except Exception as e:
        logging.exception("Error during prediction:")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
