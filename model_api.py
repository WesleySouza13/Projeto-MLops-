from fastapi import FastAPI
import joblib
import numpy as np 
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('modelo_reg_logistica')

#criando api 
class ModelInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
@app.get('/')  
def home():
    teste = 'testando'
    return teste
    
@app.post('/features')
def features(data: ModelInput):
        
    return data.model_dump()
@app.post('/inference')
def inference(data: ModelInput):
    data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    predict = model.predict(data)
    return {f'a previsao é: {int(predict[0])}'}