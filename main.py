
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

# ======================
# LOAD MODEL + FEATURES
# ======================
model = joblib.load("loan_model.pkl")
features = joblib.load("features.pkl")

# ======================
# APP INIT
# ======================
app = FastAPI(title="Loan Approval API 🚀")

# ======================
# INPUT FORMAT
# ======================
class LoanInput(BaseModel):
    name: str
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

# ======================
# HOME ROUTE
# ======================
@app.get("/")
def home():
    return {"status": "API Running 🚀"}

# ======================
# PREDICT + AUTO SAVE
# ======================
@app.post("/predict")
def predict(data: LoanInput):

    df = pd.DataFrame([data.dict()])

    name = df["name"][0]
    df = df.drop("name", axis=1)

    # encoding same as training
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    result = "Approved 🚀" if prob > 0.6 else "Rejected ❌"

    # ======================
    # AUTO SAVE LOG
    # ======================
    log = pd.DataFrame([{
        "name": name,
        "probability": prob,
        "result": result,
        "time": datetime.now()
    }])

    log.to_csv("predictions_log.csv", mode="a", header=False, index=False)

    return {
        "name": name,
        "probability": float(prob),
        "result": result
    }
