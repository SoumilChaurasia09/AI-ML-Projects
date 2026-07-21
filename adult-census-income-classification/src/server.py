import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Define FastAPI app
app = FastAPI(
    title="Adult Census Income Classification API",
    description="API to predict if an individual earns over $50K per year.",
    version="1.0"
)

# Load the trained model pipeline (preprocessor + classifier)
MODEL_PATH = os.path.join("models", "best_income_classifier_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run src/train.py first.")

print(f"Loading model pipeline from {MODEL_PATH}...")
pipeline = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Define Pydantic request body for incoming data
class CensusInput(BaseModel):
    age: int = Field(..., ge=17, le=100, description="Age of the individual")
    workclass: str = Field(..., description="Employment category")
    fnlwgt: int = Field(200000, ge=1000, description="Final weight / population weight estimate")
    education_num: int = Field(..., alias="education-num", ge=1, le=16, description="Years of education completed")
    marital_status: str = Field(..., alias="marital-status", description="Marital status")
    occupation: str = Field(..., description="Occupation type")
    relationship: str = Field(..., description="Household relationship status")
    race: str = Field(..., description="Race of the individual")
    sex: str = Field(..., description="Gender ('Male' or 'Female')")
    capital_gain: int = Field(..., alias="capital-gain", ge=0, description="Capital gains recorded")
    capital_loss: int = Field(..., alias="capital-loss", ge=0, description="Capital losses recorded")
    hours_per_week: int = Field(..., alias="hours-per-week", ge=1, le=100, description="Hours worked per week")
    native_country: str = Field(..., alias="native-country", description="Country of origin")

    # Allow both camelCase, hyphenated, and snake_case field resolution
    class Config:
        populate_by_name = True

@app.post("/predict")
def predict_income(data: CensusInput):
    """
    Takes census demographic info, preprocesses it, and runs the XGBoost classification model.
    """
    try:
        # Convert request Pydantic model to a dictionary with original dataset column names
        # Map fields to match exactly with the features the model was trained on
        input_dict = {
            'age': data.age,
            'workclass': data.workclass,
            'fnlwgt': data.fnlwgt,
            'education-num': data.education_num,
            'marital-status': data.marital_status,
            'occupation': data.occupation,
            'relationship': data.relationship,
            'race': data.race,
            'sex': data.sex,
            'capital-gain': data.capital_gain,
            'capital-loss': data.capital_loss,
            'hours-per-week': data.hours_per_week,
            'native-country': data.native_country
        }
        
        # Create a single-row Pandas DataFrame
        df = pd.DataFrame([input_dict])
        
        # Run prediction and probability using the combined pipeline
        prediction = int(pipeline.predict(df)[0])
        probability = float(pipeline.predict_proba(df)[0][1])  # Probability of class 1 (>50K)
        
        # Analyze contribution factors roughly (positive vs negative drivers)
        # We can look at the inputs to provide dynamic feedback in the UI
        reasons = []
        if data.education_num >= 13:
            reasons.append("Higher education level (Bachelor's or above) strongly supports higher income.")
        elif data.education_num < 9:
            reasons.append("Fewer years of education completed is statistical drag on income potential.")
            
        if data.capital_gain > 5000:
            reasons.append("Significant capital gains are strong indicator of high income.")
            
        if "Married-civ-spouse" in data.marital_status:
            reasons.append("Marital status (Married with civilian spouse) is strongly associated with high income classification.")
            
        if data.hours_per_week > 45:
            reasons.append("High weekly work hours (>45 hrs) positively influences high income.")
        elif data.hours_per_week < 30:
            reasons.append("Part-time hours (<30 hrs) makes high income classification less likely.")
            
        if len(reasons) == 0:
            reasons.append("The prediction is driven by standard demographic weights.")

        return {
            "prediction": prediction,
            "prediction_label": ">50K" if prediction == 1 else "<=50K",
            "probability": probability,
            "reasons": reasons
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

# Expose model metadata for the frontend
@app.get("/model-info")
def get_model_info():
    return {
        "model_type": "XGBoost Classifier",
        "features": [
            "age", "workclass", "fnlwgt", "education-num", "marital-status", 
            "occupation", "relationship", "race", "sex", "capital-gain", 
            "capital-loss", "hours-per-week", "native-country"
        ],
        "metrics": {
            "logistic_regression": {"accuracy": 0.8037, "f1_score": 0.6696, "auc": 0.9036},
            "random_forest": {"accuracy": 0.8127, "f1_score": 0.6854, "auc": 0.9144},
            "tuned_xgboost": {"accuracy": 0.8380, "f1_score": 0.7126, "auc": 0.9281}
        }
    }

# Serve static files
os.makedirs("static", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """Serves the index.html frontend page."""
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://127.0.0.1:8080...")
    uvicorn.run("server:app", host="127.0.0.1", port=8080, reload=True)
