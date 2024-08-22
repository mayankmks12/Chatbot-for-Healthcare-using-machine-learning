import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import pandas as pd

# Step 1: Load Pre-trained NLP Model
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: inquiry, appointment, symptom
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Train a Symptom Assessment Model (for demonstration purposes)
# Example symptom assessment dataset
data = {
    'text': ["I have a headache and fever", "My throat is sore", "I feel dizzy and weak"],
    'label': [0, 1, 2]  # 0: General, 1: ENT, 2: Neurology
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train a logistic regression model
symptom_model = LogisticRegression()
symptom_model.fit(X, y)

# Save the model and vectorizer for later use
joblib.dump(symptom_model, 'symptom_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Step 3: Create a Flask App for the Chatbot
app = Flask(__name__)

# Load the trained symptom model and vectorizer
symptom_model = joblib.load('symptom_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['message']
    
    # Use the NLP model to classify the type of query
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    response = ""
    
    if predicted_class == 0:  # Inquiry
        response = "How can I assist you with your healthcare needs?"
    elif predicted_class == 1:  # Appointment
        response = "Sure, let's schedule an appointment. Please provide your preferred date and time."
    elif predicted_class == 2:  # Symptom Assessment
        # Use the symptom model to predict possible conditions
        symptom_vector = vectorizer.transform([user_input])
        symptom_prediction = symptom_model.predict(symptom_vector)[0]
        
        if symptom_prediction == 0:
            response = "It seems like a general health issue. Please consult your primary care physician."
        elif symptom_prediction == 1:
            response = "It might be an ENT issue. Would you like to see a specialist?"
        elif symptom_prediction == 2:
            response = "It could be related to neurology. I recommend seeing a neurologist."
    
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

import requests

url = "http://127.0.0.1:5000/chatbot"
data = {"message": "I have a sore throat and headache."}
response = requests.post(url, json=data)

print(response.json())

