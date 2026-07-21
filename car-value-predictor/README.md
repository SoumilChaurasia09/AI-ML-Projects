# AutoValuate AI — Used Car Resale Value Predictor

AutoValuate AI is a premium, machine-learning-powered used car resale value estimation dashboard. It uses a **Random Forest Regressor** trained on the **CarDekho V3 dataset** (~7,900 cleaned rows) to predict the resale price of a vehicle based on 10 structural and mechanical specifications.

The application features a sleek, custom-designed **Emerald Forest Green & Gold** web interface with responsive layouts, real-time value retention gauges, currency toggles (INR/USD), and numerical count-up animations.

---

## Key Features
- **10 ML Features**: Predicts market value using advanced inputs like **Engine Displacement (cc)**, **Max Power (bhp)**, **Fuel Economy (mileage)**, **Seating Capacity**, and usage statistics.
- **Direct Valuation**: No need to know the original showroom price. Input the car's specifications and the model handles the valuation directly.
- **Premium UI / Visual Design**:
  - HSL-powered **Emerald Green & Gold** styling theme.
  - Fully responsive layout for Desktop, Tablet, and Mobile.
  - Animated gauge displaying the estimated **Value Retention Percentage**.
  - Dynamic currency converter between **INR (Lakhs)** and **USD ($)**.
  - Dark Mode and Light Mode support with a responsive switch.

---

## Machine Learning & Performance
The regression model is built using `scikit-learn`'s `RandomForestRegressor`. Categorical inputs (Fuel, Transmission, Owner type, Seller type) are ordinally mapped, and string-based units are dynamically parsed and cleaned in the dataset preprocessing pipeline.

### Model Evaluation Metrics:
- **R² Score (Coefficient of Determination)**: `0.9837` (the model explains over 98.3% of the price variance)
- **Mean Absolute Error (MAE)**: `0.6109 Lakhs` (~$733 USD)
- **Root Mean Squared Error (RMSE)**: `1.0659 Lakhs`

---

## Tech Stack
- **ML & Data Pipeline**: Python, `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Backend API**: Python `Flask` (REST API and static asset router)
- **Frontend UI**: Vanilla HTML5, CSS3 (CSS Variables, Flexbox/Grid, transitions), Vanilla JavaScript (fetch API, SVG circle gauge, count-up animations)

---

## Project Structure
```text
car-value-predictor/
├── data/                      # Local dataset storage (Git-ignored)
│   └── car_details_v3.csv
├── models/                    # Model binary and metadata configurations
│   ├── car_price_model.joblib # Trained Random Forest model binary (Git-ignored)
│   └── model_metadata.json    # Feature lists, mappings, and evaluation metrics
├── src/
│   ├── train.py               # Preprocessing and training script
│   ├── server.py              # Flask server and REST API
│   └── web/                   # Web application frontend assets
│       ├── index.html         # Main dashboard page
│       ├── style.css          # Theme styles (Emerald Green & Gold)
│       └── app.js             # Form triggers, gauges, and fetch API communications
├── .gitignore                 # Standard exclusions (caches, dataset, joblib binaries)
├── requirements.txt           # Python library requirements
└── README.md                  # Project documentation
```

---

## How to Get Started

### 1. Clone the Project & Install Dependencies
First, install the required packages in your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the pipeline script. It will automatically download the raw used-car dataset from GitHub, preprocess the data, evaluate the model, and serialize the trained binary:
```bash
python src/train.py
```

### 3. Start the Web Server
Launch the Flask backend server:
```bash
python src/server.py
```

### 4. Open the Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```
*Modify vehicle features (Age, Kilometers Driven, Engine CC, BHP, Seating, Fuel, etc.) and hit **Estimate Resale Value** to watch your valuation report compile!*
