from fastapi import FastAPI
# from typing import Union
import joblib
import pandas as pd
from pydantic import BaseModel


pipeline = joblib.load('./models/best_gs_model.joblib')
#encoder = joblib.load('')

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
	Age :int

# Define a route at the root web address ("/")
@app.get("/")
def status_check():
	return {"Status": "API is online!!!"}




@app.post("/prediction")
def predict_sepssis(data:patient_features):

    df = pd.DataFrame([data.model_dump()])
    #pipeline.predict(df)

	#return {"All docus": "API Documentation"}

