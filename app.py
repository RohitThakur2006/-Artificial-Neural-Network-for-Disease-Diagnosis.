from flask import Flask, request, render_template
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load model and data
with open("model/embeddings.pkl", "rb") as f:
    model_data = pickle.load(f)

embeddings = model_data['embeddings']
disease_names = model_data['disease_names']
source_urls = model_data['source_urls']
sentences = model_data['sentences']

# Load the sentence transformer model
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

def predict_disease(user_input):
    user_embedding = transformer_model.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_idx = np.argmax(similarities)
    top_score = similarities[top_idx]

    return disease_names[top_idx], source_urls[top_idx], sentences[top_idx], top_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    weight = float(request.form['weight'])
    height_cm = float(request.form['height'])
    #converting to bmi
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obesity"

    disease, url, sentence, score = predict_disease(symptoms)

    prediction_text = f"<strong>Disease:</strong> {disease}<br>" \
                      f"<strong>Matched Sentence:</strong> {sentence}<br>" \
                      f"<strong>Confidence Score:</strong> {score:.2f}<br>" \
                      f"<strong>Source:</strong> <a href='{url}' target='_blank'>{url}</a>"
    bmi_text = f"<strong>Your BMI:</strong> {bmi:.1f} ({bmi_category})"
    return render_template('index.html', prediction_text=prediction_text, bmi_text=bmi_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
