from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel



xgb_pipeline = joblib.load("../models/best_gs_model.joblib")
encoder = joblib.load('../models/label_encoder.joblib')


'''to run the API, run this line which is based on the api directory then 'uvicorn api(api python file):app(instance of fast API) --reload' '''
# Create a FastAPI application
app = FastAPI()

class patient_features(BaseModel):
	PRG :float
	PL :float
	PR :float
	SK :float
	TS :float
	M11 :float
	BD2 :float
	Age :float
	Insurance :float

# Define a route at the root web address ("/")
@app.get("/")
def status_check():
	return {"Status": "API is online!!!"}




@app.post("/xgb_model")
def predict_sepssis(data:patient_features):

    df = pd.DataFrame([data.model_dump()])
    xgb_pipeline.predict(df)
    prediction = int(prediction[0])
    probability = xgb_pipeline.predict_proba(df)

    prediction = encoder.inverse_transform([prediction])[0]
 
    if prediction == 'Negative':
            probability= f'{round(probability[0][0], 2)*100}%'
    else:
            probability = f'{round(probability[0][1], 2)*100}%'
 
    return {"prediction": prediction, "probability": probability}


