from flask import Flask, request, render_template 
from src.preprocess import load_data, clean_data
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

MODEL_PATH = 'models/price_pipeline.pkl'

pipeline = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    df = load_data()
    df = clean_data(df)
    
    car_models = sorted(df["name"].dropna().unique())
    chassis_codes = sorted(df["chassis_code"].dropna().unique())

    return render_template("index.html", car_models=car_models, chassis_codes=chassis_codes)

@app.route("/predict", methods =["POST"])
def predict():
    
    df = load_data()
    df = clean_data(df)

    has_sunroof = int("has_sunroof" in request.form)
    has_leather = int("has_leather" in request.form)
    has_navigation = int("has_navigation" in request.form)
    has_alloy_wheels = int("has_alloy_wheels" in request.form)
    has_4wd = int("has_4wd" in request.form)
    has_airbag = int("has_airbag" in request.form)
    has_abs = int("has_abs" in request.form)
    has_camera = int("has_camera" in request.form)

    options = (has_sunroof + has_leather + has_navigation + has_alloy_wheels + has_4wd + has_airbag + has_abs + has_camera)

    input_df = pd.DataFrame([{
        "name": request.form.get("name"),
        "chassis_code": request.form.get("chassis_code"),
        "year": int(request.form.get("year")),
        "mileage_km": int(request.form.get("mileage_km")),
        "engine_cc": int(request.form.get("engine_cc")),
        "has_sunroof": has_sunroof,
        "options": options,
        "has_leather": has_leather,
        "has_navigation": has_navigation,
        "has_alloy_wheels": has_alloy_wheels,
        "has_4wd": has_4wd,
        "has_airbag": has_airbag,
        "has_abs": has_abs,
        "has_camera": has_camera,
        "fuel": request.form.get("fuel"),
        "transmission": request.form.get("transmission"),
        "vehicle_type": request.form.get("vehicle_type"),

    }])
    prediction = pipeline.predict(input_df)[0]
    prediction = np.expm1(prediction)
    price = f"${prediction:,.2f}" 
    
    car_models = sorted(df["name"].dropna().unique())
    chassis_codes = sorted(df["chassis_code"].dropna().unique())   
    
    return render_template("index.html", car_models=car_models, chassis_codes=chassis_codes, price_usd=price)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    

    

    
