import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json

def download_dataset(url, dest_path):
    print(f"Downloading new V3 dataset from {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset successfully saved to {dest_path}")
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

def preprocess_and_train():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "car_details_v3.csv")
    dataset_url = "https://raw.githubusercontent.com/fenago/datasets/master/Car%20details%20v3.csv"

    if not os.path.exists(csv_path):
        download_dataset(dataset_url, csv_path)

    # Load dataset
    df = pd.read_csv(csv_path)
    print("\nDataset Info:")
    print(df.info())

    # Preprocessing
    # 1. Drop rows with nulls in key engineering columns
    df = df.dropna(subset=['mileage', 'engine', 'max_power', 'seats'])

    # 2. Parse string columns to floats
    # Clean Mileage (remove 'kmpl' or 'km/kg')
    df['mileage'] = df['mileage'].str.split().str[0].astype(float)
    
    # Clean Engine (remove 'CC')
    df['engine'] = df['engine'].str.split().str[0].astype(float)
    
    # Clean Max Power (remove 'bhp', handle spaces, convert to numeric)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').str.strip()
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    
    # Drop rows where max_power parsing failed (e.g. empty or non-numeric strings)
    df = df.dropna(subset=['max_power'])

    # 3. Calculate Age (current year 2026)
    df['Age'] = 2026 - df['year']

    # 4. Map target to Lakhs (divide by 100,000)
    df['selling_price_lakhs'] = df['selling_price'] / 100000.0

    # 5. Category mappings
    fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3}
    seller_mapping = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
    transmission_mapping = {'Manual': 0, 'Automatic': 1}
    owner_mapping = {
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }

    df['fuel'] = df['fuel'].map(fuel_mapping)
    df['seller_type'] = df['seller_type'].map(seller_mapping)
    df['transmission'] = df['transmission'].map(transmission_mapping)
    df['owner'] = df['owner'].map(owner_mapping)

    # Validate mapping didn't produce nulls
    df = df.dropna(subset=['fuel', 'seller_type', 'transmission', 'owner'])

    features = ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'Age']
    X = df[features]
    y = df['selling_price_lakhs']

    print(f"\nFinal dataset features shape: {X.shape}")
    print("Features used for training:", features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    print("\nTraining Random Forest Regressor on 10 features...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nModel Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} Lakhs")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} Lakhs")

    # Save model and metadata
    model_path = os.path.join(models_dir, "car_price_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    metadata = {
        'features': features,
        'fuel_mapping': fuel_mapping,
        'seller_mapping': seller_mapping,
        'transmission_mapping': transmission_mapping,
        'owner_mapping': owner_mapping,
        'evaluation': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
    }
    
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    preprocess_and_train()
