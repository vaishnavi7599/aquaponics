from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# Create a Flask app that serves the frontend from a separate directory
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Create necessary directories
os.makedirs('backend/model', exist_ok=True)
os.makedirs('backend/files', exist_ok=True)

# Path constants
MODEL_PATH = 'backend/model/random_forest_model.pkl'
SCALER_PATH = 'backend/model/scaler.pkl'
DATA_PATH = 'files/training_farm_data.csv'

# Load the trained model and scaler
def load_model():
    try:
        # Load the model and scaler
        model = pickle.load(open(MODEL_PATH, 'rb'))
        scaler = pickle.load(open(SCALER_PATH, 'rb'))
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Train and save the model if it doesn't exist
def train_and_save_model():
    try:
        # Load data
        data = pd.read_csv(DATA_PATH)
        
        # Encode location
        LE = LabelEncoder()
        data['location'] = LE.fit_transform(data['location'])
        
        # Split data
        X = data.drop(columns=['yield'])
        y = data['yield']
        
        # Scale features
        scaler = StandardScaler()
        Scaled_X = scaler.fit_transform(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(Scaled_X, y, test_size=0.2, random_state=40)
        
        # Initialize and train Random Forest model
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        
        # Save model and scaler
        pickle.dump(rf, open(MODEL_PATH, 'wb'))
        pickle.dump(scaler, open(SCALER_PATH, 'wb'))
        
        print("Model trained and saved successfully!")
        return rf, scaler
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

# Route to serve the HTML page
@app.route('/')
def home():
    return app.send_static_file('index.html')

# Route to handle prediction requests
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json['input']
        
        # Load model and scaler
        model, scaler = load_model()
        
        # If model doesn't exist, train and save it
        if model is None or scaler is None:
            model, scaler = train_and_save_model()
            
        if model is None or scaler is None:
            return jsonify({'error': 'Failed to load or train model'}), 500
        
        # Transform input data
        transformed_data = scaler.transform([input_data])
        
        # Make prediction
        prediction = model.predict(transformed_data)
        
        # Return prediction result
        return jsonify({'result': f"Predicted yield: {prediction[0]:.2f} kgs"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if model exists, if not train and save it
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    
    app.run(debug=True)