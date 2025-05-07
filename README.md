# ML Model Deployment: Spam Classifier API with Flask, Docker, and CI/CD

## Project Overview

This project demonstrates the end-to-end process of building a Machine Learning model and deploying it as a containerized web API with integrated SQL logging and an automated CI/CD pipeline. The project focuses on text classification for Spam detection.

## Problem Statement

Effectively identifying and filtering spam messages is crucial for managing digital communication. This project aims to build an automated system that can predict whether a given message is spam, accessible via a standard web API for integration into applications or services.

## Dataset

* **Name:** SMS Spam Collection Dataset
* **Source:** Available on Kaggle (https://www.kaggle.com/uciml/sms-spam-collection-dataset).
* **Content:** Contains a collection of SMS messages labeled as 'ham' (not spam) or 'spam'.
* **Size:** Contains 5572 messages, split into training and testing sets (80/20 split).
* **Challenge:** The dataset has a significant class imbalance, with many more 'ham' messages than 'spam' messages.

## Methodology

This project integrates several components to create a deployable ML service:

1.  **ML Model Development:**
    * Loaded and preprocessed the SMS message data.
    * Used **TF-IDF Vectorization** to convert message text into numerical features (5000 features).
    * Trained a **Logistic Regression** model using scikit-learn, incorporating `class_weight='balanced'` to address class imbalance during training.
    * Saved the trained model, the TF-IDF vectorizer, and the label encoder using `joblib`.
2.  **Flask API Development:**
    * Created a Python Flask application (`app.py`).
    * The app loads the saved ML model and vectorizer into memory on startup.
    * Defined a `/predict` **REST API endpoint** that accepts POST requests with a JSON body containing the message text (e.g., `{"message": "Your text"}`).
    * The API uses the loaded model to preprocess the input message, make a prediction, and return a JSON response with the predicted label ('ham' or 'spam') and the prediction probabilities.
3.  **SQL Logging:**
    * Integrated **SQLite** database logging into the Flask API.
    * For each incoming prediction request, the API logs the original message, predicted label, and probabilities into a `predictions` table in a `prediction_logs.db` file.
4.  **Dockerization:**
    * Created a `Dockerfile` to package the Flask application, saved ML model files, the Python environment, and all dependencies (`requirements.txt`).
    * The Docker image provides a portable and consistent environment for running the API.
5.  **Git and Gitlab:**
    * Used **Git** for version control to track code changes.
    * Hosted the project repository on **Gitlab.com**.
6.  **CI/CD Pipeline (Gitlab CI):**
    * Configured a Continuous Integration/Continuous Deployment pipeline using **`.gitlab-ci.yml`**.
    * The pipeline automatically triggers whenever code is pushed to the `main` branch.
    * The pipeline includes a job that performs both the **`docker build`** (creating the Docker image) and **`docker push`** (sending the image to the Gitlab Container Registry) within the same execution, ensuring the image is available for deployment.

## Results

The trained Logistic Regression model achieved strong performance on the test set:

* **Test Accuracy:** **0.9821**
* **Confusion Matrix:**
    * True Ham, Predicted Ham (TN): 958
    * True Ham, Predicted Spam (FP): 8 (Low false alarms)
    * True Spam, Predicted Ham (FN): 12 (Successfully identified most spam)
    * True Spam, Predicted Spam (TP): 137
* **Classification Report:**
    * **Spam Precision:** 0.94 (Out of messages predicted as spam, 94% were actually spam)
    * **Spam Recall:** 0.92 (The model found 92% of all actual spam messages)
    * Spam F1-score: 0.93

These metrics indicate a highly effective spam classifier with a good balance of minimizing false alarms while still catching most spam.

## Evaluation Metrics Explained

* **Accuracy:** Overall proportion of correct predictions.
* **Confusion Matrix:** Breakdown of correct (True Positives/Negatives) and incorrect (False Positives/Negatives) predictions for each class.
* **Precision:** Accuracy specifically within the set of *predicted* positive cases. Important for minimizing false alarms (especially for spam).
* **Recall (Sensitivity):** Proportion of actual positive cases that were correctly identified. Important for not missing positive cases (especially for spam).
* **F1-score:** Harmonic mean of Precision and Recall, providing a single score that balances both.

## Live Demo

A live version of this Spam Classifier API is deployed and publicly accessible.

* **Platform:** Render.com (using the Free Tier)
* **URL:** `https://spam-classifier-api-qsr3.onrender.com`
    * *(Note: The base URL may change if you rename the Render service; always use the URL provided by Render).*
* **Access Method:** Send a **POST** request to the `/predict` endpoint with JSON data containing your message. The full prediction endpoint URL is `https://spam-classifier-api-qsr3.onrender.com/predict`.

You can test the live API using `curl` (or `Invoke-WebRequest` in PowerShell) or a simple Python `requests` script.

**Example using `curl` (from a standard terminal like Git Bash):**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"message": "Claim your prize now!"}' [https://spam-classifier-api-qsr3.onrender.com/predict](https://spam-classifier-api-qsr3.onrender.com/predict)