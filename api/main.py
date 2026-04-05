from fastapi import FastAPI
import joblib
import pandas as pd
from typing import List

# Création de l'API
app = FastAPI()

# Chargement de notre modèle
model = joblib.load("./models/xgb_model_detection_fraude.pkl")

@app.get("/")
def home():
    return {"message": "L'API de détection de fraude est en route"}

# Création de la route API
@app.post("/prediction")
def prediction(transactions: List[dict]):
    df = pd.DataFrame(transactions)

    predictions = model.predict(df)
    probabilitesfraude = model.predict_proba(df)[:, 1]

    #On retourne la liste des résultats
    resultats = []

    #On associe chaque prédiction à sa probabilité (zip) et on affiche le résultat
    for prediction, probabilitefraude in  zip(predictions, probabilitesfraude):
        resultats.append({
            "prediction": int(prediction),
            "probabilitedefraude": float(probabilitefraude)
        })
    return resultats
