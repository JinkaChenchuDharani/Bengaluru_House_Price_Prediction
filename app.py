from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
app = Flask(__name__)

# Load the model and columns.json
model = pickle.load(open('banglore_home_price_model', 'rb'))
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Prepare input data
    loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1
    input_data = np.zeros(len(data_columns))  # this should match columns.json

    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk

    if loc_index >= 0:
        input_data[loc_index] = 1

    # Log the number of features for debugging
    print(f"Number of input features: {len(input_data)}")
    print(f"Model expects {model.coef_.shape[0]} features")

    # Check for feature mismatch
    if len(input_data) != model.coef_.shape[0]:
        return render_template('index.html', prediction_text="Feature mismatch error. Please check input data.")

    # Predict the price using the model
    predicted_price = model.predict([input_data])[0]

    return render_template('index.html', prediction_text=f"Predicted House Price: {predicted_price:,.2f} lakhs")


if __name__ == "__main__":
    app.run(debug=True)
