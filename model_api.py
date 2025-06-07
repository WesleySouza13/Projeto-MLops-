from fastapi import FastAPI
import joblib
import numpy as np 

app = FastAPI()
model = joblib.load('modelo_reg_logistica')

#criando api 
class model_api():
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    @app.get('/')
    def home():
        teste = 'testando'
        return teste
    
    @app.get('/features')
    def features(sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    ):
        
        features = {
            sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
    
        }
        return features
    @app.post('/inference')
    def inference():
        data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        predict = model.predict(data)
        return {f'a previsao é: {int(predict[0])}'}