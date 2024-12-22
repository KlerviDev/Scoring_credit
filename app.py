from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle et les encodeurs
model = joblib.load('Gradient_boosting_model.pkl')
le_sex = joblib.load('sex_encoder.pkl')
le_saving = joblib.load('saving_accounts_encoder.pkl')
le_checking = joblib.load('checking_account_encoder.pkl')
housing_columns = joblib.load('housing_columns.pkl')  # Liste des colonnes one-hot pour Housing
purpose_columns = joblib.load('purpose_columns.pkl')  # Liste des colonnes one-hot pour Purpose

# Création de l'application FastAPI
app = FastAPI()

class PredictionInput(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Credit_amount: float
    Duration: int
    Purpose: str

# Endpoint pour vérifier que l'API fonctionne
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API FastAPI pour prédire le risque de crédit !"}

# Endpoint pour la prédiction
@app.post("/predict/")
def predict(data: PredictionInput):
    try:
        # Appliquer le LabelEncoder aux variables catégorielles
        data.Sex = le_sex.transform([data.Sex])[0]
        data.Saving_accounts = le_saving.transform([data.Saving_accounts])[0]
        data.Checking_account = le_checking.transform([data.Checking_account])[0]

        # Appliquer OneHotEncoder pour 'Housing' et 'Purpose'
        housing_one_hot = [1 if col == f"Housing_{data.Housing}" else 0 for col in housing_columns]
        purpose_one_hot = [1 if col == f"Purpose_{data.Purpose}" else 0 for col in purpose_columns]

        # Préparer les données en un tableau numpy pour la prédiction
        input_data = np.array([[ 
            data.Age,
            data.Sex,
            data.Job,
            data.Saving_accounts,
            data.Checking_account,
            *housing_one_hot,  # Ajouter les colonnes one-hot pour Housing
            data.Credit_amount,
            data.Duration,
            *purpose_one_hot  # Ajouter les colonnes one-hot pour Purpose
        ]])

        # Debug : Vérifie la taille des données générées
        print("Shape of input_data:", input_data.shape)

        # Faire la prédiction avec le modèle pré-entraîné
        prediction = model.predict(input_data)

        # Retourner le résultat sous forme de texte pour l'utilisateur
        risk = "Good" if prediction[0] == 1 else "Bad"  # Adapter selon ton encodage

        return {"risk": risk}
  
    except Exception as e:
        return {"error": str(e)}

