from flask import Flask, request, jsonify
import uuid
from datetime import datetime
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re

app = Flask(__name__)

# Required Custom Classes

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]


class StatisticalFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            features.append([
                len(text),
                text.count("!"),
                text.count("?"),
                sum(word.isupper() for word in text.split())
            ])
        return np.array(features)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "mvr_logistic_classifier.joblib")

model = joblib.load(model_path)
print("Model loaded successfully.")



def compute_urgency_score(text):
    prediction = model.predict([text])[0]
    return prediction



@app.route("/tickets", methods=["POST"])
def create_ticket():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Missing required field: text"}), 400

    ticket_id = str(uuid.uuid4())
    text = data["text"]

    prediction = compute_urgency_score(text)

    ticket = {
        "ticket_id": ticket_id,
        "text": text,
        "prediction": prediction,
        "status": "processed",
        "created_at": datetime.utcnow().isoformat()
    }

    return jsonify(ticket), 200


if __name__ == "__main__":
    app.run(debug=True)