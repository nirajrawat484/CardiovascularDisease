#API
'''
Types of Requests
-------------------
Get -> Read/select
post -> Create/ Insert / Send
put -> Update
Delete -> Delete/remove
'''
from fastapi import FastAPI
from app.schema import CardioSchema
from app.model import load_logistic_model
import pandas as pd
#fast API Object
app = FastAPI()
model, scaler = load_logistic_model()

# API Endpoints / Requests
@app.get('/')
def home():
    return 'Welcome to Cardiovascular Disease Prediction'


@app.post('/predict-cardio-logistic')
def predict_cardio_logistic(data:CardioSchema):
    input_data = pd.DataFrame([
        #Accept data as JSON format
        data.model_dump()
        
    ])
    Input_scaler = scaler.transform(input_data)
    prediction = model.predict(Input_scaler)[0] # 0 or 1
    return {
        'prediction': int(prediction),
        'Status': 'Likely to be Healthy ' if prediction == 0 else 'Likely to be Unhelathy'
    }
