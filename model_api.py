from fastapi import FastAPI
import joblib
import numpy as np 
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('modelo_reg_logi.pkl')

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
        if model is None:
            return {"erro": 'modelo nao carregado'}
    
        try:
            input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
            prediction = model.predict(input_array)
            return {"previsao": int(prediction[0])}
        except Exception as e:
            return {"erro": str(e)}