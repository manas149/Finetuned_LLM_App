from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
model_path = "./models/fine_tuned_model"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = sentiment_pipeline(text)
    return jsonify({"text": text, "sentiment": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
