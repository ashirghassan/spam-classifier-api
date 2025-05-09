# Step 2.2: Write the Flask Application (app.py)

# Set up SQLite Database in app.py
from flask import Flask, request, jsonify, render_template 
import joblib
import os
import pandas as pd
import numpy as np
import sqlite3 # Import the SQLite library


# --- Configuration ---
# Define the paths to your saved model files
# Assumes the files are in the same directory as this app.py script
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory of this script
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_classifier_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
DATABASE_NAME = 'prediction_logs.db' # Name of the SQLite database file
DATABASE_PATH = os.path.join(MODEL_DIR, DATABASE_NAME) # Full path to the database file


# Function to create the predictions table if it doesn't exist
def create_predictions_table():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_message TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                probability_ham REAL,
                probability_spam REAL
            )
        ''')
        conn.commit()
        print(f"Database table 'predictions' checked/created successfully at {DATABASE_PATH}.")
    except sqlite3.Error as e:
        print(f"Database error during table creation: {e}")
    finally:
        if conn:
            conn.close()

# Call the function to ensure the table exists when the app starts
create_predictions_table()



# --- Load Model and Preprocessing Tools ---
# Load the saved model, vectorizer, and label encoder when the application starts
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Model, vectorizer, and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    # Exit or handle error appropriately if files can't be loaded
    model = None
    vectorizer = None
    label_encoder = None


# --- Initialize Flask App ---
app = Flask(__name__)

# --- Basic Route for Frontend ---
@app.route('/')
def index():
    # This tells Flask to look for 'index.html' in a 'templates' folder
    return render_template('index.html')

# --- Define Prediction Endpoint ---
@app.route('/predict', methods=['POST'])


# Corrected predict_spam function
def predict_spam():
    # Ensure model files were loaded
    if model is None or vectorizer is None or label_encoder is None:
        print("Error: Model files not loaded in Flask app.") # Log error server-side
        return jsonify({'error': 'Model files not loaded. Check server logs.'}), 500 # Internal Server Error

    # Get the data from the incoming request
    # Expecting JSON data like: {"message": "Your message text here"}
    data = request.json # Get JSON data from the request body

    # Check if the 'message' key is present in the request data
    if data is None or 'message' not in data or not isinstance(data['message'], str):
        print(f"Error: Invalid input data: {data}") # Log error server-side
        return jsonify({'error': 'Invalid input. Please provide JSON with a "message" key containing text.'}), 400 # Bad Request

    # Extract the message text
    message = data['message']

    # --- Preprocess the message ---
    # The vectorizer expects a list of strings, even for a single message
    try:
        message_vectorized = vectorizer.transform([message])
        # Check if the vectorized output is valid (e.g., has expected shape)
        if message_vectorized.shape[1] != 5000: # Check against max_features used in vectorizer
             print(f"Warning: Vectorized message shape {message_vectorized.shape} unexpected.")
             # Continue, but this could indicate a problem
    except Exception as e:
        print(f"Error vectorizing message '{message}': {e}") # Log error server-side
        return jsonify({'error': f'Error vectorizing message: {e}'}), 500 # Internal Server Error

    # --- Make Prediction ---
    try:
        # predict() returns an array of shape (n_samples,), get the prediction for the first sample
        prediction_index = model.predict(message_vectorized)[0]

        # predict_proba() returns an array of shape (n_samples, n_classes), get probabilities for the first sample
        prediction_proba_all = model.predict_proba(message_vectorized)[0] # e.g., [prob_ham, prob_spam]

        # Get the predicted class name ('ham' or 'spam') using the index
        predicted_label = label_encoder.inverse_transform([prediction_index])[0]

        # Get the probability for the predicted class index
        predicted_proba_value = prediction_proba_all[prediction_index]

        # Get probabilities for ham and spam using the encoder's mapping
        prob_ham = prediction_proba_all[label_encoder.transform(['ham'])[0]]
        prob_spam = prediction_proba_all[label_encoder.transform(['spam'])[0]]


    except Exception as e:
        print(f"Error during prediction for message '{message}': {e}") # Log error server-side
        return jsonify({'error': f'Error making prediction: {e}'}), 500 # Internal Server Error


    # --- Return Response ---
    # Return the prediction and probabilities as a JSON response
    response = {
        'message': message, # Include the original message in the response
        'predicted_label': predicted_label,
        'predicted_probability': float(predicted_proba_value), # Convert to float for JSON
        'probability_ham': float(prob_ham), # Convert to float
        'probability_spam': float(prob_spam)  # Convert to float
    }

    # --- Step 2.6 (Optional/Advanced): Add SQL Logging ---
    # Add logging logic here in a later step

    print(f"Prediction made for message '{message}': {predicted_label} ({predicted_proba_value:.4f})") # Log prediction server-side
    # --- Log Prediction to Database ---
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (input_message, predicted_label, probability_ham, probability_spam)
            VALUES (?, ?, ?, ?)
        ''', (message, predicted_label, prob_ham, prob_spam)) # Use values from the prediction result
        conn.commit()
        print(f"Logged prediction to database: '{message}' -> {predicted_label}") # Log server-side
    except sqlite3.Error as e:
        print(f"Database error during logging prediction: {e}")
        # We don't return a 500 here, as the prediction itself was successful,
        # but we log the database error.
    finally:
        if conn:
            conn.close()
    return jsonify(response), 200 # Return JSON response with status code 200 (OK)

# --- Basic Route (Optional) ---
# You can add a simple route to check if the server is running

# --- Run the Flask Development Server ---
if __name__ == '__main__':
    # Run the app in debug mode and listen on all public IPs (needed for Docker)
    # debug=True is useful for development, but disable in production
    # host='0.0.0.0' makes the server accessible from outside the container
    app.run(debug=False, host='0.0.0.0') # Changed debug to False