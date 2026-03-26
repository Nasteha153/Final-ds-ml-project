from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from utils import prepare_features_from_raw

app = Flask(__name__)
CORS(app)

# Load models
MODELS = {
    "lr": joblib.load("models/lr_model.joblib"),
    "rf": joblib.load("models/rf_model.joblib"),
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Car Price Prediction API",
        "endpoints": {
            "POST /predict?model=lr|rf": {
                "expects_json": {
                    "year": "int",
                    "km_driven": "number",
                    "fuel": "Petrol|Diesel",
                    "seller_type": "Dealer|Individual",
                    "transmission": "Manual|Automatic",
                    "owner": "int"
                }
            }
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    # 1) Choose model
    choice = (request.args.get("model") or "").lower()
    if choice not in MODELS:
        return jsonify({"error": "Use model=lr or model=rf"}), 400

    model = MODELS[choice]

    # 2) Get JSON input
    data = request.get_json(silent=True) or {}

    required = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
    missing = [k for k in required if k not in data]

    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # 3) Prepare features
        x_new = prepare_features_from_raw(data)

        # 4) Predict
        pred = float(model.predict(x_new)[0])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "model": "linear_regression" if choice == "lr" else "random_forest",
        "input": data,
        "predicted_price": round(pred, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)