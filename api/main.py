from fastapi import FastAPI
import joblib
import pandas as pd

# Création de l'API
app = FastAPI()

# Chargement de notre modèle
model = joblib.load("./models/xgb_model_detection_fraude.pkl")

@app.get("/")
def home():
    return {"message": "L'API de détection de fraude est en route"}

# Création de la route API
@app.post("/prediction")
def prediction(data: dict):
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probabilitefraude = model.predict_proba(df)[0][1]

    return {
        "prediction" : int(prediction) ,
        "probabilitefraude" : float(probabilitefraude)
    }
