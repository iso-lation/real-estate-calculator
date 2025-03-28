from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)
# load model from model directory
model_path = os.path.join(os.path.dirname(__file__),'..', 'models', 'price_estimator_model.pkl')
model = joblib.load(model_path)
print("Model path:", model_path)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        features = [
            data['Gr Liv Area'],
            data['Bedroom AbvGr'],
            data['Full Bath'],
            data['Year Built'],
            data['Garage Cars'],
            data['Lot Area']
        ]
    except KeyError:
        return jsonify({'error': 'Missing one or more input values'}), 400

    prediction = model.predict([np.array(features)])
    estimated_price = max(0, prediction[0])
    estimated_price = "${:,.2f}".format(prediction[0])

    return jsonify({'estimated_price': estimated_price})

if __name__ == '__main__':
    app.run(debug=True)