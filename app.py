from flask import Flask, jsonify, render_template, request
from src.predictor import CropPredictor
from src.validation import validate_payload

app = Flask(__name__)
predictor = CropPredictor()

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": predictor.pipeline is not None})

@app.post("/api/predict")
def api_predict():
    try:
        payload = validate_payload(request.get_json(silent=True))
        result = predictor.predict(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Request body too large."}), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
