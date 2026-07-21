import os
import sys
import joblib
import json
from flask import Flask, request, jsonify, send_from_directory

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")

# Load model and metadata
model_path = os.path.join(MODELS_DIR, "car_price_model.joblib")
metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

try:
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run src/train.py first.", file=sys.stderr)
        model = None
        metadata = {}
    else:
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("V2 Model and metadata successfully loaded.")
except Exception as e:
    print(f"Error loading model: {str(e)}", file=sys.stderr)
    model = None
    metadata = {}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not trained or loaded. Please run train.py first."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Validate inputs
        required_fields = ['year', 'km_driven', 'fuel_type', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract values
        year = int(data['year'])
        km_driven = int(data['km_driven'])
        fuel_type = data['fuel_type']
        seller_type = data['seller_type']
        transmission = data['transmission']
        owner = data['owner']
        mileage = float(data['mileage'])
        engine = float(data['engine'])
        max_power = float(data['max_power'])
        seats = float(data['seats'])

        # Preprocess features
        # 1. Calculate Age (2026 - year)
        age = 2026 - year
        if age < 0:
            return jsonify({"error": "Manufacturing year cannot be in the future."}), 400

        # 2. Get category mappings
        fuel_mapping = metadata.get('fuel_mapping', {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3})
        seller_mapping = metadata.get('seller_mapping', {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2})
        transmission_mapping = metadata.get('transmission_mapping', {'Manual': 0, 'Automatic': 1})
        owner_mapping = metadata.get('owner_mapping', {
            'Test Drive Car': 0,
            'First Owner': 1,
            'Second Owner': 2,
            'Third Owner': 3,
            'Fourth & Above Owner': 4
        })

        # 3. Map categories
        fuel_mapped = fuel_mapping.get(fuel_type)
        seller_mapped = seller_mapping.get(seller_type)
        trans_mapped = transmission_mapping.get(transmission)
        owner_mapped = owner_mapping.get(owner)

        if fuel_mapped is None:
            return jsonify({"error": f"Invalid fuel_type: {fuel_type}. Must be one of {list(fuel_mapping.keys())}"}), 400
        if seller_mapped is None:
            return jsonify({"error": f"Invalid seller_type: {seller_type}. Must be one of {list(seller_mapping.keys())}"}), 400
        if trans_mapped is None:
            return jsonify({"error": f"Invalid transmission: {transmission}. Must be one of {list(transmission_mapping.keys())}"}), 400
        if owner_mapped is None:
            return jsonify({"error": f"Invalid owner value: {owner}. Must be one of {list(owner_mapping.keys())}"}), 400

        # Feature array order:
        # ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'Age']
        features = [
            km_driven,
            fuel_mapped,
            seller_mapped,
            trans_mapped,
            owner_mapped,
            mileage,
            engine,
            max_power,
            seats,
            age
        ]

        # Make prediction (returns Selling Price in Lakhs)
        prediction = model.predict([features])[0]
        predicted_price = max(0.05, prediction) # Minimum 5,000 INR

        # Synthetic Original Showroom Price calculation using a realistic exponential decay model:
        # Retention = 0.87 ^ Age (Assuming ~13% yearly depreciation)
        # Therefore, Original Price = Predicted Price / (0.87 ^ Age)
        # We also enforce that original price must be at least the predicted price.
        depreciation_rate = 0.87
        retention_ratio = max(0.1, depreciation_rate ** age)
        original_price = predicted_price / retention_ratio
        
        # Upper limit constraints: let's verify if original price is logically sound.
        # If age is 0, retention is 100%.
        if age == 0:
            original_price = predicted_price
            retention_ratio = 1.0

        depreciation = original_price - predicted_price
        retained_percentage = retention_ratio * 100

        response = {
            "predicted_price_lakhs": round(predicted_price, 2),
            "original_price_lakhs": round(original_price, 2),
            "depreciation_lakhs": round(depreciation, 2),
            "value_retention_percentage": round(retained_percentage, 1),
            "estimated_age_years": age,
            "currency_conversion": {
                "approx_rate_usd_per_lakh_inr": 1200.0,
                "predicted_price_usd": round(predicted_price * 1200, 2),
                "original_price_usd": round(original_price * 1200, 2),
                "depreciation_usd": round(depreciation * 1200, 2)
            }
        }

        return jsonify(response)

    except ValueError as ve:
        return jsonify({"error": f"Invalid input data type: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "metadata": metadata.get('evaluation', {})
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting V2 server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
