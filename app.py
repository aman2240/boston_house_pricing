import pickle
from flask import Flask, jsonify, url_for, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print("Received Data:", data)

        # Convert data to NumPy array
        new_data = np.array(list(data.values())).reshape(1, -1)
        print("Input Array:", new_data)

        # Apply scaling
        new_data_scaled = scalar.transform(new_data)

        # Predict output
        output = regmodel.predict(new_data_scaled)
        print("Prediction:", output[0])

        return jsonify({'prediction': float(output[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
