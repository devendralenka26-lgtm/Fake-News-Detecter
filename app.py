from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

# Load trained model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        news_text = data.get("news", "")

        if not news_text.strip():
            return jsonify({"error": "No text provided"}), 400

        clean_text = preprocess(news_text)
        vec = vectorizer.transform([clean_text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max() * 100

        return jsonify({
            "prediction": pred,
            "confidence": round(prob, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
    #end of project
    
